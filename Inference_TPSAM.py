from datetime import datetime
from tqdm import tqdm
import argparse
import os
import logging
import glob
import random
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn

from segment_anything.modeling import Sam
from segment_anything import sam_model_registry
from TPSAM import TPSAM, PromptGenerator

from torch.utils.data import Dataset, DataLoader
from utils_V2 import get_Dice, get_IoU, get_Recall, get_mean_Dice_IoU_Recall

import warnings
warnings.filterwarnings("ignore")

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
	
def parse_argus():
    parser = argparse.ArgumentParser(description="Testing SAM")
    # image, mask, prior path
    parser.add_argument("--image_paths", type=str, help='subfolders images, training npy files')
    parser.add_argument("--mask_paths", type=str, help='subfolders masks, training npy files')
    parser.add_argument("--prior_paths", type=str, help='subfolders priors, training npy files')

    parser.add_argument("--dataset_name", type=str, default="MMAC-FS", choices=["MMAC-FS", "MMAC-LC", "PALM-Atrophy"])
    parser.add_argument("--task_name", type=str, default="TPSAM-ViT-B", choices=["SAM-ViT-B","TPSAM-ViT-B"])
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--checkpoint", type=str, 
                        default='/data/home/litingyao/project/SAM/TextPromptSAM/work_dir/MMAC-LC/TPSAM-ViT-B_20240909-202322/model_save_path/tpsam_model_best.pth')
    parser.add_argument("--work_dir", type=str, default="./work_dir")
    # parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    
    parser.add_argument("--use_amp", action="store_true", default=False, help="use_amp")

    parser.add_argument('--cuda', type=bool, default=True, help='whether to use cuda')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=str, default='2', help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()
    args = add_args_inference(args)

    return args

def add_args_inference(args):
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.store_name = args.task_name+'_'+run_id
    args.experiment_dir = os.path.join(args.work_dir, args.dataset_name, args.store_name)
    args.logging_dir = os.path.join(args.experiment_dir, 'logs')
    os.makedirs(args.logging_dir, exist_ok=True)

    args.log_txt = os.path.join(args.logging_dir, args.store_name+'.log')
    args.py_name = os.path.basename(__file__).split('.')[0]

    args.seg_dir = os.path.join(args.experiment_dir, 'Image_MaskGT_and_MaskPred_npz')
    args.vis_dir = os.path.join(args.experiment_dir, 'Image_MaskGT_and_MaskPred_Vis')
    os.makedirs(args.seg_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)
    return args

def _seed_torch(args):
	r"""
	Sets custom seed for torch

	Args:
		- seed : Int

	Returns:
		- None

	"""
	import random
	seed = args.seed
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	if args.cuda:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		if device.type == 'cuda':
			torch.cuda.manual_seed(seed)
			torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
		else:
			raise EnvironmentError("GPU device not found")
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def visualize_image_mask_predbest_pred_component(npz_file):
    data = np.load(npz_file)
    image = data['image']
    mask_pred_best = data['mask_pred']
    mask_gt = data['mask_gt']
    name = npz_file.split('/')[-1].split('.')[0]
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))

    axs[0].imshow(image)
    axs[0].set_title(name)
    axs[0].axis('off')

    axs[1].imshow(mask_gt.squeeze(), cmap='gray')
    axs[1].set_title("MaskGT")
    axs[1].axis('off')

    axs[2].imshow(mask_pred_best.squeeze(), cmap='gray')
    axs[2].set_title("MaskPred_Best")
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(args.vis_dir,name+'.png'))
    plt.close()

def visual_all(args):
    logging.info("Begin Visualization!")
    npz_dir = args.seg_dir
    for i in sorted(os.listdir(npz_dir)):
        npz_file = os.path.join(npz_dir,i)   
        visualize_image_mask_predbest_pred_component(npz_file)
    logging.info("Finish Visualization!")

def metrics_all(args):
    logging.info("Begin Metrics!")
    IoU_list = []
    Dice_list = []
    Recall_list = []

    npz_dir = args.seg_dir
    for i in sorted(os.listdir(npz_dir)):
        npz_file = os.path.join(npz_dir,i)  
        name = npz_file.split('/')[-1].split('.')[0]

        data = np.load(npz_file)
        image = data['image']
        mask_pred = data['mask_pred'].squeeze()
        mask_gt = data['mask_gt'].squeeze()

        i_IoU = get_IoU(pred=mask_pred, gt=mask_gt)
        i_Dice = get_Dice(pred=mask_pred, gt=mask_gt)
        i_Recall = get_Recall(pred=mask_pred,gt=mask_gt)

        logging.info("Name: {}, i_IoU: {:.4f}, i_Dice: {:.4f}, i_Recall: {:.4f}".format(name, i_IoU, i_Dice, i_Recall))

        IoU_list.append(i_IoU)
        Dice_list.append(i_Dice)
        Recall_list.append(i_Recall)

    
    mIoU = np.nanmean(IoU_list)
    mDice = np.nanmean(Dice_list)
    mRecall = np.nanmean(Recall_list)

    logging.info("mIoU: {:.4f}, mDice: {:.4f}, mRecall: {:.4f}".format(mIoU, mDice, mRecall))
    logging.info("Finish Metrics!")


def inference(args):
    logging.basicConfig(filename=args.log_txt, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s -%(message)s')

    logging.info("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device) if torch.cuda.is_available() else 'cpu'

    test_dataset = NpyDataset(
        image_paths = args.image_paths,
        mask_paths = args.mask_paths,
        prior_paths = args.prior_paths,
        mode='train',  
    )
	
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
	
    logging.info("========= Dataset Loading Done ===============")
	
    vit_b_checkpoint = "/data/home/litingyao/project/SAM/LearnablePromptSAM/sam_checkpoints/sam_vit_b_01ec64.pth"
    sam_model = sam_model_registry['vit_b'](checkpoint=vit_b_checkpoint)

    prompt_generator = PromptGenerator(num_classes=1, dense_channels=256, sparse_channels=256)

    tpsam_model = TPSAM(
            sam_model=sam_model,
            prompt_generator=prompt_generator)

    tpsam_checkpoint_path = '/data/home/litingyao/project/SAM/TextPromptSAM/work_dir/MMAC-LC/TPSAM-ViT-B_20240909-202322/model_save_path/tpsam_model_best.pth'
    tpsam_checkpoint = torch.load(tpsam_checkpoint_path)
    tpsam_model.load_state_dict(tpsam_checkpoint['model'])
    tpsam_model = tpsam_model.to(device)
    logging.info("========= Model Loading Done ===============")
	
    tpsam_model.eval()
    with torch.no_grad():
        for step, (image, mask, prior, temp_name) in enumerate(tqdm(test_dataloader)):
            image, mask, prior = image.to(device), mask.to(device), prior.to(device)
            
            pred_mask = tpsam_model(image, prior)
            pred_mask = pred_mask.detach().cpu().numpy()
            pred_mask = pred_mask / pred_mask.max() 
            pred_mask = (pred_mask > 0.5).astype(np.uint8)

            image_np = np.transpose(image.cpu().numpy().squeeze(0),(1,2,0))
            mask_np = mask.detach().cpu().numpy()

            npz_save_path = os.path.join(args.seg_dir, str(temp_name[0])+'.npz')
            np.savez_compressed(npz_save_path, image=image_np, mask_pred=pred_mask, mask_gt=mask_np)
    logging.info("Inference Test!")

if __name__ == '__main__':
    args = parse_argus()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    _seed_torch(args)
    inference(args)
    metrics_all(args)
    visual_all(args)
    print('Done!')