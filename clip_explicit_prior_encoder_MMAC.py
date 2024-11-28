import torch
import clip

from PIL import Image, ImageOps
from torch import nn
from contextlib import contextmanager
import numpy as np
import os
import pandas as pd

def min_max_normalize(tensor, min_val=0.0, max_val=1.0):
    # Get the minimum and maximum values from the tensor
    min_tensor = tensor.min()
    max_tensor = tensor.max()
    
    # Perform min-max normalization
    norm_tensor = (tensor - min_tensor) / (max_tensor - min_tensor + 1e-10)
    # Scale to the desired range [min_val, max_val]
    norm_tensor = norm_tensor * (max_val - min_val) + min_val
    
    return norm_tensor

def resize_and_pad_image(input_image, target_size=(1024, 1024)):
    original_size=input_image.size
    # rescaling image, let the longer side is equal to the longer side of the target size
    ratio = min(target_size[0]/original_size[0], target_size[1]/original_size[1])
    new_size = (int(original_size[0]*ratio), int(original_size[1]*ratio))
    # rescaling
    resize_image = input_image.resize(new_size, Image.BICUBIC)
    # get num of pixels to fill
    delta_w = target_size[0]-new_size[0]
    delta_h = target_size[1]-new_size[1]
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    # filling using black color
    padded_image = ImageOps.expand(resize_image, padding, fill=(0, 0, 0))
    return padded_image

class PriorExtractor:
	def __init__(self, model_name='ViT-B/16', device=None):
		self.device = device
		self.model, self.preprocess = clip.load(model_name, device=self.device)
		self.feature_map = None

	@contextmanager
	def register_transformer_hook(self):
		def hook_fn(module, input, output):
			self.feature_map = output

		hook = self.model.visual.transformer.register_forward_hook(hook_fn)
		try:
			yield
		finally:
			hook.remove()

	def min_max_normalize(self, tensor, min_val=0.0, max_val=1.0):
		# Get the minimum and maximum values from the tensor
		min_tensor = tensor.min()
		max_tensor = tensor.max()
		
		# Perform min-max normalization
		norm_tensor = (tensor - min_tensor) / (max_tensor - min_tensor + 1e-10)
		# Scale to the desired range [min_val, max_val]
		norm_tensor = norm_tensor * (max_val - min_val) + min_val
		
		return norm_tensor

	def get_visual_features(self, image):
		with self.register_transformer_hook():
			with torch.no_grad():
				_ = self.model.encode_image(image.type(self.model.visual.conv1.weight.dtype))
		
		if self.feature_map is None:
			raise ValueError("Hook did not capture the transformer output.")

		permuted_feature_map = self.feature_map.permute(1, 0, 2)
		after_ln_post = self.model.visual.ln_post(permuted_feature_map[:, 1:, :])
		after_proj = after_ln_post @ self.model.visual.proj
		
		# print('after_proj shape:', after_proj.shape)
		return after_proj

	def compute_similarity(self, P_v, P_t):
		P_s = torch.matmul(P_v, P_t.transpose(-1, -2).unsqueeze(0))
		return P_s

	def extract_prior(self, image_path, text_prompt):
		input_image = Image.open(image_path).convert("RGB")
		resize_image = resize_and_pad_image(input_image)
		image = self.preprocess(resize_image).unsqueeze(0).to(self.device)
		# image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
		text = clip.tokenize([text_prompt]).to(self.device)

		with torch.no_grad():
			image_features = self.get_visual_features(image)
			text_features = self.model.encode_text(text)

		P_v = image_features / image_features.norm(dim=-1, keepdim=True)
		P_t = text_features / text_features.norm(dim=-1, keepdim=True)

		P_s = self.compute_similarity(P_v, P_t)
		P_s_prime = P_s.reshape(1, 14, 14)
		P_e = self.min_max_normalize(P_s_prime)

		P_e_resized = nn.functional.interpolate(P_e.unsqueeze(1), size=(64, 64), mode='bilinear').squeeze(1)
		# print('P_e_resized shape:', P_e_resized.shape)

		return P_e_resized

# image_dir = '/data/home/litingyao/project/SAM/data/1-Lesion-IDRiD-Segmentation/1. Original Images'
# save_dir = '/data/home/litingyao/project/SAM/data/IDRiD_NPY'

def get_explict_prior_npz(extractor, set_type, image_name, mask_type):
	assert set_type in ('1. Training Set', '2. Validation Set', '3. Testing Set')
	assert mask_type in ('1. Lacquer Cracks', '2. Choroidal Neovascularization', '3. Fuchs Spot')
	image_dir = '/data/home/litingyao/project/SAM/data/MMAC/2. Segmentation of Myopic Maculopathy Plus Lesions'
	save_dir = '/data/home/litingyao/project/SAM/data/MMAC_NPY'


	image_set_dir = os.path.join(image_dir, mask_type, '1. Images', set_type)
	gt_dir = os.path.join(image_dir, mask_type, '2. Groundtruths')

	image_path = os.path.join(image_set_dir, image_name)

	# if mask_type == '1. Microaneurysms':
	# 	text_description = 'tiny red dots'
	# 	suffix = '_MA'
	# elif mask_type == '2. Haemorrhages':
	# 	text_description = 'red or blotchy spots'
	# 	suffix = '_HE'
	# elif mask_type == '3. Hard Exudates':
	# 	text_description = 'yellowish-white deposits'
	# 	suffix = '_EX'
	# elif mask_type == '4. Soft Exudates':
	# 	text_description = 'fluffy, white patches'
	# 	suffix = '_SE'
	if mask_type == '3. Fuchs Spot':
		text_description = 'Pigmented grayish white scar'
	elif mask_type == '2. Choroidal Neovascularization':
		text_description = ''
	elif mask_type == '1. Lacquer Cracks':
		text_description = 'Yellowish thick linear lesions in the macula'


	ex_prior_torch = extractor.extract_prior(image_path, text_description)
	ex_prior_numpy = ex_prior_torch.cpu().numpy()

	save_set_dir = os.path.join(save_dir,  mask_type,'Prior', set_type)

	os.makedirs(save_set_dir, exist_ok=True)
	save_npy_path = os.path.join(save_set_dir, image_name.split('.')[0]+'.npy')
	np.save(save_npy_path, ex_prior_numpy)

def locate_df(set_type, mask_type):
	if mask_type == '1. Lacquer Cracks':
		if set_type == '1. Training Set':
			df = pd.read_csv('/data/home/litingyao/project/SAM/data/MMAC/2. Segmentation of Myopic Maculopathy Plus Lesions/1. Lacquer Cracks/2. Groundtruths/1. MMAC2023_Myopic_Maculopathy_Plus_Lesions_Segmentation_Training_Images_Lacquer_Cracks.csv').copy()
		elif set_type == '2. Validation Set':
			df = pd.read_csv('/data/home/litingyao/project/SAM/data/MMAC/2. Segmentation of Myopic Maculopathy Plus Lesions/1. Lacquer Cracks/2. Groundtruths/2. MMAC2023_Myopic_Maculopathy_Plus_Lesions_Segmentation_Validation_Images_Lacquer_Cracks.csv').copy()
		elif set_type == '3. Testing Set':
			df = pd.read_csv('/data/home/litingyao/project/SAM/data/MMAC/2. Segmentation of Myopic Maculopathy Plus Lesions/1. Lacquer Cracks/2. Groundtruths/3. MMAC2023_Myopic_Maculopathy_Plus_Lesions_Segmentation_Testing_Images_Lacquer_Cracks.csv').copy()
	elif mask_type == '2. Choroidal Neovascularization':
		if set_type == '1. Training Set':
			df = pd.read_csv('/data/home/litingyao/project/SAM/data/MMAC/2. Segmentation of Myopic Maculopathy Plus Lesions/2. Choroidal Neovascularization/2. Groundtruths/1. MMAC2023_Myopic_Maculopathy_Plus_Lesions_Segmentation_Training_Images_Choroidal_Neovascularization.csv').copy()
		elif set_type == '2. Validation Set':
			df = pd.read_csv('/data/home/litingyao/project/SAM/data/MMAC/2. Segmentation of Myopic Maculopathy Plus Lesions/2. Choroidal Neovascularization/2. Groundtruths/2. MMAC2023_Myopic_Maculopathy_Plus_Lesions_Segmentation_Validation_Images_Choroidal_Neovascularization.csv').copy()
		elif set_type == '3. Testing Set':
			df = pd.read_csv('/data/home/litingyao/project/SAM/data/MMAC/2. Segmentation of Myopic Maculopathy Plus Lesions/2. Choroidal Neovascularization/2. Groundtruths/3. MMAC2023_Myopic_Maculopathy_Plus_Lesions_Segmentation_Testing_Images_Choroidal_Neovascularization.csv').copy()
	elif mask_type == '3. Fuchs Spot':
		if set_type == '1. Training Set':
			df = pd.read_csv('/data/home/litingyao/project/SAM/data/MMAC/2. Segmentation of Myopic Maculopathy Plus Lesions/3. Fuchs Spot/2. Groundtruths/1. MMAC2023_Myopic_Maculopathy_Plus_Lesions_Segmentation_Training_Images_Fuchs_Spot.csv').copy()
		elif set_type == '2. Validation Set':
			df = pd.read_csv('/data/home/litingyao/project/SAM/data/MMAC/2. Segmentation of Myopic Maculopathy Plus Lesions/3. Fuchs Spot/2. Groundtruths/2. MMAC2023_Myopic_Maculopathy_Plus_Lesions_Segmentation_Validation_Images_Fuchs_Spot.csv').copy()
		elif set_type == '3. Testing Set':
			df = pd.read_csv('/data/home/litingyao/project/SAM/data/MMAC/2. Segmentation of Myopic Maculopathy Plus Lesions/3. Fuchs Spot/2. Groundtruths/3. MMAC2023_Myopic_Maculopathy_Plus_Lesions_Segmentation_Testing_Images_Fuchs_Spot.csv').copy()
	return df


if __name__ == '__main__':

	device = 'cuda:0' if torch.cuda.is_available() else 'cpu'	
	extractor = PriorExtractor(device=device)


	# train_csv = '/data/home/litingyao/project/SAM/data/1-Lesion-IDRiD-Segmentation/Train_Image_Mask.csv'
	# train_df = pd.read_csv(train_csv).copy()
	# test_csv = '/data/home/litingyao/project/SAM/data/1-Lesion-IDRiD-Segmentation/Test_Image_Mask.csv'
	# test_df = pd.read_csv(test_csv).copy()

	
	# for set_type in ['1. Training Set', '2. Validation Set', '3. Testing Set']:
	# 	for mask_type in ['1. Lacquer Cracks', '3. Fuchs Spot']:
	# 		df = locate_df(set_type=set_type, mask_type=mask_type)
				
	# 		image_list = list(df['image'])
	# 		print(image_list)

	# 		for image_name in image_list:
	# 			get_explict_prior_npz(extractor, set_type, image_name, mask_type)
	
	for set_type in ['3. Testing Set']:
		for mask_type in ['3. Fuchs Spot']:
			df = locate_df(set_type=set_type, mask_type=mask_type)
				
			image_list = list(df['image'])
			print(image_list)

			for image_name in image_list:
				get_explict_prior_npz(extractor, set_type, image_name, mask_type)