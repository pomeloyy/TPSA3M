import numpy as np
import os
import torch
from PIL import Image
import glob
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random

torch.manual_seed(2024)

class MyDataset(Dataset):
	def __init__(self, image_paths, mask_paths, prior_paths):
		self.image_paths = image_paths
		self.mask_paths = mask_paths
		self.prior_paths = prior_paths

		self.prior_path_files = sorted(
			glob.glob(os.path.join(self.prior_paths, "**/*.npy"), recursive=True)
		)
		# print('self.prior_path_files===================>')
		# print(self.prior_path_files)
		self.gt_path_files = sorted(
			glob.glob(os.path.join(self.mask_paths, "**/*.npy"), recursive=True)
		)
		# print('self.gt_path_files===================>')
		# print(self.gt_path_files)
		# # self.gt_prefix = [
		# # 	'_'.join(os.path.basename(file).split('_')[0:2]) + '.npy' 
		# # 	for file in self.gt_path_files]
		# self.gt_prefix = [
		# 	'_'.join(os.path.basename(file).split('_')) + '.npy' 
		# 	for file in self.gt_path_files]
		# print('self.gt_prefix===================>')
		# print(self.gt_prefix)
		self.gt_prefix =  [os.path.basename(file).split('.')[0] 
					 for file in self.gt_path_files]
		# self.img_path_files = [
		# 	os.path.join(self.image_paths, file)
		# 	for file in self.gt_prefix
		# 	if os.path.isfile(os.path.join(self.image_paths, file))
        # ]
		self.img_path_files = sorted(
			glob.glob(os.path.join(self.image_paths, "**/*.npy"), recursive=True)
		)

		# print('self.img_path_files===================>')
		# print(self.img_path_files)

		self.length = len(self.gt_path_files)
		# print(f"number of images: {self.length}")
	
	def __len__(self):
		return self.length
	
	def __getitem__(self, index):
		prefix_id = self.gt_prefix[index]
		image_path = self.img_path_files[index]
		mask_path = self.gt_path_files[index]
		prior_path = self.prior_path_files[index]

		# load npy image, (1024, 1024, 3), [0,1]
		image = np.load(image_path) 
		# convert the shape to (3, H, W)
		image = np.transpose(image, (2,0,1))
		assert (np.max(image)<=1.0 and np.min(image)>=0.0), "image should be normalized to [0,1]"

		# load npy mask, (1024, 1024), [0, 1]
		mask = np.load(mask_path) 
		assert (np.max(mask)==1 and np.min(mask)==0), "mask gt should be 0,1"

		# load npy explicit prior, (1, 64, 64)
		prior = np.load(prior_path) 
		# prior = prior.squeeze()

		return (
			torch.tensor(image).float(),
			torch.tensor(mask).long(),
			torch.tensor(prior).float(),
			prefix_id
		)

class NpyDataset(Dataset):
	def __init__(self, image_paths, mask_paths, prior_paths, mode='train'):
		self.image_paths = image_paths
		self.mask_paths = mask_paths
		self.prior_paths = prior_paths

		assert "1. Training Set" in self.image_paths
		if mode == 'valid':
			self.image_paths = image_paths.replace("1. Training Set", "2. Validation Set")
			self.mask_paths = mask_paths.replace("1. Training Set", "2. Validation Set")
			self.prior_paths = prior_paths.replace("1. Training Set", "2. Validation Set")
		elif mode == "test":
			self.image_paths = image_paths.replace("1. Training Set", "3. Testing Set")
			self.mask_paths = mask_paths.replace("1. Training Set", "3. Testing Set")
			self.prior_paths = prior_paths.replace("1. Training Set", "3. Testing Set")

		# print(self.image_paths)
		# print(self.mask_paths)
		# print(self.prior_paths)
		self.prior_path_files = sorted(
			glob.glob(os.path.join(self.prior_paths, "**/*.npy"), recursive=True)
		)

		self.gt_path_files = sorted(
			glob.glob(os.path.join(self.mask_paths, "**/*.npy"), recursive=True)
		)

		self.gt_prefix =  [os.path.basename(file).split('.')[0] 
					 for file in self.gt_path_files]

		self.img_path_files = sorted(
			glob.glob(os.path.join(self.image_paths, "**/*.npy"), recursive=True)
		)

		self.length = len(self.gt_path_files)
		# print(f"number of images: {self.length}")
	
	def __len__(self):
		return self.length
	
	def __getitem__(self, index):
		prefix_id = self.gt_prefix[index]
		image_path = self.img_path_files[index]
		mask_path = self.gt_path_files[index]
		prior_path = self.prior_path_files[index]

		# load npy image, (1024, 1024, 3), [0,1]
		image = np.load(image_path) 
		# convert the shape to (3, H, W)
		image = np.transpose(image, (2,0,1))
		assert (np.max(image)<=1.0 and np.min(image)>=0.0), "image should be normalized to [0,1]"

		# load npy mask, (1024, 1024), [0, 1]
		mask = np.load(mask_path) 
		assert (np.max(mask)==1 and np.min(mask)==0), "mask gt should be 0,1"

		# load npy explicit prior, (1, 64, 64)
		prior = np.load(prior_path) 
		# prior = prior.squeeze()

		return (
			torch.tensor(image).float(),
			torch.tensor(mask).long(),
			torch.tensor(prior).float(),
			prefix_id
		)
		
def show_mask(mask, ax, random_color=False):
	if random_color:
		color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
	else:
		# color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6]) # bright green
		color = np.array([0 / 255, 255 / 255, 0 / 255, 0.6])  # 60% 透明度的亮绿色
	h, w = mask.shape[-2:]
	mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
	ax.imshow(mask_image)

if __name__ == '__main__':	
	image_paths = '/data/home/litingyao/project/SAM/data/MMAC_NPY/1. Lacquer Cracks/Images/1. Training Set'
	mask_paths = '/data/home/litingyao/project/SAM/data/MMAC_NPY/1. Lacquer Cracks/Masks/1. Training Set'
	prior_paths = '/data/home/litingyao/project/SAM/data/MMAC_NPY/1. Lacquer Cracks/Prior/1. Training Set'

	dataset = NpyDataset(image_paths,mask_paths, prior_paths, mode='valid')
	print(dataset.__len__())
	# # print(idrid_dataset.__getitem__(0))

	# idrid_dataset = MyDataset(
	# 	# mask_paths = '/data/home/litingyao/project/SAM/data/IDRiD_NPY/Masks/a. Training Set/4. Soft Exudates',
	# 	# prior_paths = '/data/home/litingyao/project/SAM/data/IDRiD_NPY/Prior/a. Training Set/4. Soft Exudates'
		
	# 	# image_paths = '/data/home/litingyao/project/SAM/data/MMAC_NPY/3. Fuchs Spot/Images/1. Training Set',
	# 	# mask_paths = '/data/home/litingyao/project/SAM/data/MMAC_NPY/3. Fuchs Spot/Masks/1. Training Set',
	# 	# prior_paths = '/data/home/litingyao/project/SAM/data/MMAC_NPY/3. Fuchs Spot/Prior/1. Training Set'

	# 	# image_paths = '/data/home/litingyao/project/SAM/data/MMAC_NPY/3. Fuchs Spot/Images/2. Validation Set',
	# 	# mask_paths = '/data/home/litingyao/project/SAM/data/MMAC_NPY/3. Fuchs Spot/Masks/2. Validation Set',
	# 	# prior_paths = '/data/home/litingyao/project/SAM/data/MMAC_NPY/3. Fuchs Spot/Prior/2. Validation Set'

	# 	# image_paths = '/data/home/litingyao/project/SAM/data/MMAC_NPY/3. Fuchs Spot/Images/3. Testing Set',
	# 	# mask_paths = '/data/home/litingyao/project/SAM/data/MMAC_NPY/3. Fuchs Spot/Masks/3. Testing Set',
	# 	# prior_paths = '/data/home/litingyao/project/SAM/data/MMAC_NPY/3. Fuchs Spot/Prior/3. Testing Set'

	# 	# image_paths = '/data/home/litingyao/project/SAM/data/MMAC_NPY/1. Lacquer Cracks/Images/1. Training Set',
	# 	# mask_paths = '/data/home/litingyao/project/SAM/data/MMAC_NPY/1. Lacquer Cracks/Masks/1. Training Set',
	# 	# prior_paths = '/data/home/litingyao/project/SAM/data/MMAC_NPY/1. Lacquer Cracks/Prior/1. Training Set'

	# 	# image_paths = '/data/home/litingyao/project/SAM/data/MMAC_NPY/1. Lacquer Cracks/Images/2. Validation Set',
	# 	# mask_paths = '/data/home/litingyao/project/SAM/data/MMAC_NPY/1. Lacquer Cracks/Masks/2. Validation Set',
	# 	# prior_paths = '/data/home/litingyao/project/SAM/data/MMAC_NPY/1. Lacquer Cracks/Prior/2. Validation Set'

	# 	image_paths = '/data/home/litingyao/project/SAM/data/MMAC_NPY/1. Lacquer Cracks/Images/3. Testing Set',
	# 	mask_paths = '/data/home/litingyao/project/SAM/data/MMAC_NPY/1. Lacquer Cracks/Masks/3. Testing Set',
	# 	prior_paths = '/data/home/litingyao/project/SAM/data/MMAC_NPY/1. Lacquer Cracks/Prior/3. Testing Set'

	# )

	# print(idrid_dataset.__len__())
	# # print(idrid_dataset.__getitem__(0))
	# np.random.seed(2023)
	# image_mask_prior_dir = '/data/home/litingyao/project/SAM/data/MMAC/Image_Mask_Prior_Vis/LacquerCracks'
	# os.makedirs(image_mask_prior_dir, exist_ok=True)

	# idrid_se_dataloader = DataLoader(idrid_dataset, batch_size=4, num_workers=4, shuffle=False)
	# for i, (image, mask, prior, name_temp) in enumerate(idrid_se_dataloader):
	# 	print(image.shape, mask.shape, prior.shape, name_temp) # torch.Size([4, 3, 1024, 1024]) torch.Size([4, 1024, 1024]) torch.Size([4, 1, 64, 64]) ('IDRiD_01.npy', 'IDRiD_02.npy', 'IDRiD_03.npy', 'IDRiD_04.npy')
	# 	# show the example
		
	# 	# # idx = random.randint(0, 4)
	# 	for idx in range(len(name_temp)):
	# 		_, axs = plt.subplots(1, 3, figsize=(12, 10))
	# 		axs[0].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
	# 		axs[1].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
	# 		show_mask(mask[idx].cpu().numpy(), axs[1])
	# 		axs[2].imshow(prior[idx].cpu().numpy().squeeze(),cmap='coolwarm')
			
	# 		axs[0].axis("off")
	# 		axs[1].axis("off")
	# 		axs[2].axis("off")

	# 		# set title
	# 		axs[1].set_title(name_temp[idx])
	# 		# plt.show()
	# 		plt.subplots_adjust(wspace=0.01, hspace=0)
	# 		plt.savefig(os.path.join(image_mask_prior_dir, name_temp[idx]+'_coolwarm.png'), bbox_inches="tight")
	# 		plt.close()

	# 	for idx in range(len(name_temp)):
	# 		_, axs = plt.subplots(1, 3, figsize=(12, 10))
	# 		axs[0].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
	# 		axs[1].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
	# 		show_mask(mask[idx].cpu().numpy(), axs[1])
	# 		axs[2].imshow(prior[idx].cpu().numpy().squeeze(),cmap='coolwarm_r')
			
	# 		axs[0].axis("off")
	# 		axs[1].axis("off")
	# 		axs[2].axis("off")

	# 		# set title
	# 		axs[1].set_title(name_temp[idx])
	# 		# plt.show()
	# 		plt.subplots_adjust(wspace=0.01, hspace=0)
	# 		plt.savefig(os.path.join(image_mask_prior_dir, name_temp[idx]+'_coolwarm_r.png'), bbox_inches="tight")
	# 		plt.close()

	# # for i, (image, mask, prior, name_temp) in enumerate(idrid_se_dataloader):
	# # 	print(image.shape, mask.shape, prior.shape, name_temp) # torch.Size([4, 3, 1024, 1024]) torch.Size([4, 1024, 1024]) torch.Size([4, 1, 64, 64]) ('IDRiD_01.npy', 'IDRiD_02.npy', 'IDRiD_03.npy', 'IDRiD_04.npy')
	# # 	# show the example
	# # 	_, axs = plt.subplots(1, 3, figsize=(12, 10))
	# # 	idx = random.randint(0, 4)
	# # 	axs[0].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
	# # 	axs[1].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
	# # 	show_mask(mask[idx].cpu().numpy(), axs[1])
	# # 	axs[2].imshow(prior[idx].cpu().numpy().squeeze(),cmap='coolwarm_r')
		
	# # 	axs[0].axis("off")
	# # 	axs[1].axis("off")
	# # 	axs[2].axis("off")

	# # 	# set title
	# # 	axs[1].set_title(name_temp[idx])
	# # 	# plt.show()
	# # 	plt.subplots_adjust(wspace=0.01, hspace=0)
	# # 	plt.savefig("./data_sanitycheck_MMAC.png", bbox_inches="tight")
	# # 	plt.close()
	# # 	break