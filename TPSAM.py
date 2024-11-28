import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from segment_anything.modeling import Sam
from segment_anything import sam_model_registry
from contextlib import contextmanager

class TPSAM(nn.Module):
	def __init__(
			self,
			sam_model,
			# image_encoder,
			# mask_decoder,
			# prompt_encoder,
			prompt_generator,
			injector_layer=None
	):
		super().__init__()
		self.image_encoder = Injector_SAM(sam_model, injector_layer)
		self.mask_decoder = sam_model.mask_decoder
		self.prompt_encoder = sam_model.prompt_encoder
		self.prompt_generator = prompt_generator

		# freeze prompt encoder
		for param in self.prompt_encoder.parameters():
			param.requires_grad = False
	
	def forward(self, image, Pe):
		image_embeddings = self.image_encoder(image, Pe)
		# print('=======forward in TPSAM, image_embedding.shape:==>', image_embeddings.shape)
		# print('=======forward in TPSAM, Pe.shape:==>', Pe.shape)
		dense_embeddings, sparse_embeddings = self.prompt_generator(image_embeddings, Pe)
		# print('=======forward in TPSAM, sparse_embeddings.shape:==>', sparse_embeddings.shape)
		# print('=======forward in TPSAM, dense_embeddings.shape:==>', dense_embeddings.shape)

		low_res_masks, _ = self.mask_decoder(
			image_embeddings = image_embeddings, # (B, 256, 64, 64)
			image_pe = self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
			sparse_prompt_embeddings = sparse_embeddings, # (B, 2, 256)?  # (B, num_classes, 256)?
			dense_prompt_embeddings = dense_embeddings, # (B, 256, 64, 64)? # (B, num_classes, 256, 64, 64)?
			multimask_output = False,
		)
		# print('=======forward in TPSAM, low_res_masks.shape:==>', low_res_masks.shape)
		ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
		)
		# print('=======forward in TPSAM, ori_res_masks.shape:==>', ori_res_masks.shape)

		return ori_res_masks

class Injector_SAM(nn.Module):
	def __init__(self, sam_model: Sam, injector_layer=None):
		super(Injector_SAM, self).__init__()
		self.sam_model = sam_model

		if injector_layer:
			self.injector_layer = injector_layer
		else:
			self.injector_layer = list(range(len(sam_model.image_encoder.blocks)))

		num_features = sam_model.image_encoder.blocks[0].attn.qkv.in_features
		self.injectors = nn.ModuleList([
			PriorAlignedInjector(in_channels=num_features, hidden_dim=num_features) for i in self.injector_layer
		])

		# Freeze SAM encoder weights except for the injector layers
		for param in sam_model.image_encoder.parameters():
			param.requires_grad = False
		
		# Unfreeze the parameters of the injector
		for i in self.injector_layer:
			for param in self.injectors[i].parameters():
				param.requires_grad = True

	def forward(self, x, Pe):
		# print('forward in Injector_SAM before patch_embed.shape:==>', x.shape)
		x = self.sam_model.image_encoder.patch_embed(x)
		# print('forward in Injector_SAM after patch_embed.shape:==>', x.shape)
		# print('forward in Injector_SAM Prior input Pe.shape:==>', Pe.shape)
		for i, block in enumerate(self.sam_model.image_encoder.blocks):
			x = block(x)
			if i in self.injector_layer:
				x = self.injectors[i](x, Pe)
		x = x.permute(0,3,1,2)
		x = self.sam_model.image_encoder.neck(x)
		return x

class PriorAlignedInjector(nn.Module):
	def __init__(self, in_channels, hidden_dim, kernel_size=1, stride=4):
		super(PriorAlignedInjector, self).__init__()
		self.hidden_dim = hidden_dim
		# Define the projection layers
		self.query_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size, stride=stride)
		self.key_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size, stride=stride)
		self.value_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size, stride=stride)
		
		# Learnable parameter for blending ratio
		self.gamma = nn.Parameter(torch.zeros(1))

	def forward(self, Fi, Pe):
		# Element-wise multiplication to get F_act_i
		Fi = Fi.permute(0,3,1,2)
		# Pe = Pe.permute(0,3,1,2)

		F_act_i = Fi * Pe  # Element-wise multiplication
		# Project into query, key, and value
		F_Q_i = self.query_conv(Fi)  # (batch_size, hidden_dim, Hs/4, Ws/4)
		F_K_i = self.key_conv(F_act_i)
		F_V_i = self.value_conv(F_act_i)
		
		# Flatten spatial dimensions for matrix multiplication
		B, C, H, W = F_Q_i.shape
		F_Q_i = F_Q_i.view(B, C, -1)  # (batch_size, hidden_dim, H*W)
		F_K_i = F_K_i.view(B, C, -1)  # (batch_size, hidden_dim, H*W)
		F_V_i = F_V_i.view(B, C, -1)  # (batch_size, hidden_dim, H*W)

		# Compute attention scores
		attention = torch.bmm(F_Q_i.transpose(1, 2), F_K_i) / (self.hidden_dim ** 0.5)  # (batch_size, H*W, H*W)
		attention = F.softmax(attention, dim=-1)
		
		# Apply attention to the value
		F_prime_i = torch.bmm(attention, F_V_i.transpose(1, 2))  # (batch_size, H*W, hidden_dim)
		F_prime_i = F_prime_i.transpose(1, 2).view(B, C, H, W)  # Reshape back to (batch_size, hidden_dim, H, W)
		
		# Upsample back to original resolution
		F_prime_i = F.interpolate(F_prime_i, size=(Fi.size(2), Fi.size(3)), mode='bilinear', align_corners=False)
		# Residual connection and blending with gamma
		F_prime_i = self.gamma * F_prime_i + Fi  # Residual connection
		F_prime_i = F_prime_i.permute(0,2,3,1)
		return F_prime_i

class PromptGenerator(nn.Module):
	def __init__(self, num_classes, dense_channels, sparse_channels):
		super(PromptGenerator, self).__init__()
		self.num_classes = num_classes
		self.conv_dense = nn.Conv2d(dense_channels, dense_channels, kernel_size=1)
		self.linear_sparse = nn.Linear(4096, sparse_channels)

	def forward(self, Fe, Pe):
		B, Ce, He, We = Fe.shape  # Batch size, Channels, Height, Width
		# print("Fe.shape:==>", Fe.shape) # (4, 256, 64, 64)
		# print('Pe.shape:==>', Pe.shape) # (4, 1, 64, 64)
		Fp = Fe * Pe 
		# print('Fp.shape:==>', Fp.shape) # (4, 256, 64, 64)

		# Dense projection
		Ed = self.conv_dense(Fp)
		# print('Ed.shape:==>', Ed.shape)

		# Sparse projection
		Fp_reshape = Fp.view(B, Ce, He*We)
		# print('Fci_p_reshape.shape:==>', Fp_reshape.shape)
		Es = self.linear_sparse(Fp_reshape)
		# print('Es.shape:==>', Es.shape)
		return Ed, Es

class ClassSpecificPromptGenerator(nn.Module):
	def __init__(self, num_classes, dense_channels, sparse_channels):
		super(ClassSpecificPromptGenerator, self).__init__()
		self.num_classes = num_classes
		self.conv_dense = nn.Conv2d(dense_channels, dense_channels, kernel_size=1)
		self.linear_sparse = nn.Linear(4096, sparse_channels)
	
	def forward(self, Fe, Pe):
		# Step1: Reshape Fe
		B, Ce, He, We = Fe.shape # Batch size, Channels, Height, Width
		print('Fe.shape:==>', Fe.shape) # (4, 256, 64, 64)
		# Step2: Interaction with prior feature
		# Fe = Fe.permute(0, 2, 3, 1)
		# print('Fe.shape:==>', Fe.shape)
		print('Pe.shape:==>', Pe.shape) # (4, 1, 64, 64)
		# Pe = Pe.permute(0, 2, 3, 1)
		# print('Pe.shape:==>', Pe.shape)
		Fp = Fe * Pe 
		print('Fp.shape:==>', Fp.shape) # (4, 256, 64, 64)
		# Step3: Replicate Fp for c times
		Fp_rep = Fp.unsqueeze(1).repeat(1, self.num_classes, 1, 1, 1)
		print('Fp_rep.shape:==>', Fp_rep.shape) # (4, 2, 256, 64, 64)

		# Step4: Select category-specific channels for class ci
		Fp_class_specific = Fp_rep[:,:,:,:,:]

		# Step5: Project to get dense and sparse embeddings
		Ed_list = []
		Es_list = []

		for ci in range(self.num_classes):
			Fci_p = Fp_class_specific[:,ci,:,:,:]
			print('Fci_p.shape:==>', Fci_p.shape) # （4, 256, 64, 64）
			# Fci_p = Fci_p.permute(0,3,1,2) # Change back to (B, Ce, He, We)
			# print('Fci_p.shape:==>', Fci_p.shape)
			
			# Dense projection
			Ed = self.conv_dense(Fci_p)
			print('Ed.shape:==>', Ed.shape)
			Ed_list.append(Ed)

			# Sparse projection
			Fci_p_reshape = Fci_p.view(B, Ce, He*We)
			print('Fci_p_reshape.shape:==>', Fci_p_reshape.shape)
			Es = self.linear_sparse(Fci_p_reshape)
			print('Es.shape:==>', Es.shape)
			Es_list.append(Es)

		Ed = torch.stack(Ed_list, dim=1)
		Es = torch.stack(Es_list, dim=1)

		return Ed, Es


if __name__ == '__main__':
	torch.manual_seed(42)
	vit_b_checkpoint = "/data/home/litingyao/project/SAM/LearnablePromptSAM/sam_checkpoints/sam_vit_b_01ec64.pth"
	sam_model = sam_model_registry['vit_b'](checkpoint=vit_b_checkpoint)

	# prompt_generator = ClassSpecificPromptGenerator(num_classes=1, 
	# 												dense_channels=256,
	# 												sparse_channels=256)

	prompt_generator = PromptGenerator(num_classes=1, 
													dense_channels=256,
													sparse_channels=256)

	tpsam_model = TPSAM(
		# image_encoder=sam_model.image_encoder,
		# mask_decoder=sam_model.mask_decoder,
		# prompt_encoder=sam_model.prompt_encoder,
		sam_model=sam_model,
		prompt_generator=prompt_generator
				)

	print(
		"Number of total paramters: ",
		sum(p.numel() for p in tpsam_model.parameters())
	) # 116,111,420 116M

	print(
		"Number of trainable paramters: ",
		sum(p.numel() for p in tpsam_model.parameters() if p.requires_grad)
	) # 26,434,288 26M

	print(
		"Number of trainable paramters in image_encoder: ",
		sum(p.numel() for p in tpsam_model.image_encoder.parameters() if p.requires_grad)
	) # 25,319,664 25M

	print(
		"Number of trainable paramters in prompt_generator: ",
		sum(p.numel() for p in tpsam_model.prompt_generator.parameters() if p.requires_grad)
	) # 1,114,624 1M

	print(
		"Number of trainable paramters in mask_decoder: ",
		sum(p.numel() for p in tpsam_model.mask_decoder.parameters() if p.requires_grad)
	) # 4,058,340 4M

	# Test if tpsam works
	images = torch.rand((4, 3, 1024, 1024))
	masks = torch.randint(0, 2, (4, 1024, 1024))
	priors = torch.randn((4, 1, 64, 64))

	y = tpsam_model(images, priors)
	print('y.shape:==>', y.shape) # y.shape:==> torch.Size([4, 1, 1024, 1024])
	tpsam_pred = y.squeeze(1)
	print('tpsam_pred.shape:==>', tpsam_pred.shape) # tpsam_pred.shape:==> torch.Size([4, 1024, 1024])

	# test if need squeeze
	train_pred = tpsam_pred.detach().cpu().numpy().argmax(1).squeeze()
	train_label = masks.detach().cpu().numpy().squeeze()

	print('train_pred.shape:==>',train_pred.shape) # train_pred.shape:==> (4, 1024)
	print('train_label.shape:==>',train_label.shape) # train_label.shape:==> (4, 1024, 1024)