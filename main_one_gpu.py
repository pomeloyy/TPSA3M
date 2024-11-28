from datetime import datetime
from tqdm import tqdm
import argparse
import os
import matplotlib.pyplot as plt
import logging
import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from segment_anything.modeling import Sam
from segment_anything import sam_model_registry
from TPSAM import TPSAM,PromptGenerator
from dataset import NpyDataset,DataLoader
import monai
from utils import compute_dice_coefficient, compute_binary_recall, compute_binary_iou

import warnings
warnings.filterwarnings("ignore")

def parse_argus():
    parser = argparse.ArgumentParser(description="Traning TP-SAM")
    # image, mask, prior path
    parser.add_argument("-i", "--image_paths", type=str, help='subfolders images, training npy files')
    parser.add_argument("-m", "--mask_paths", type=str, help='subfolders masks, training npy files')
    parser.add_argument("-p", "--prior_paths", type=str, help='subfolders priors, training npy files')

    parser.add_argument("--dataset_name", type=str, default="MMAC-FS", choices=["MMAC-FS", "MMAC-LC", "PALM-Atrophy"])
    parser.add_argument("--task_name", type=str, default="TPSAM-ViT-B")
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--checkpoint", type=str, 
                        default="/data/home/litingyao/project/SAM/LearnablePromptSAM/sam_checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--work_dir", type=str, default="./work_dir")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)

    # optimizer parameters
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay(default:0.01)")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate (absolute lr)")

    parser.add_argument("--use_amp", action="store_true", default=False, help="use_amp")
    parser.add_argument("--resume", type=str, default="", help="resuming training from checkpoint")

    parser.add_argument('--cuda', type=bool, default=True, help='whether to use cuda')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=str, default='1', help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    args = add_args(args)

    return args

def add_args(args):
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.store_name = args.task_name+'_'+run_id
    args.experiment_dir = os.path.join(args.work_dir, args.dataset_name, args.store_name)
    args.model_save_path = os.path.join(args.experiment_dir, 'model_save_path')
    os.makedirs(args.model_save_path, exist_ok=True)

    tensorboard_logdir = os.path.join(args.experiment_dir, 'tensorboard_logdir')
    os.makedirs(tensorboard_logdir, exist_ok=True)

    args.tsbd_log = os.path.join(tensorboard_logdir, args.store_name)
    args.logging_dir = os.path.join(args.experiment_dir, 'logs')
    os.makedirs(args.logging_dir, exist_ok=True)

    args.log_txt = os.path.join(args.logging_dir, args.store_name+'.log')
    args.py_name = os.path.basename(__file__).split('.')[0]
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

def logging_model_info(tpsam_model):
    logging.info(
    "Number of total paramters: {:.5f}M".format(sum(p.numel() for p in tpsam_model.parameters())/1e6)
    ) # 116,111,420 116M
    logging.info(
        "Number of trainable paramters: {:.5f}M".format(sum(p.numel() for p in tpsam_model.parameters() if p.requires_grad)/1e6)
        ) # 26,434,288 26M
    logging.info(
        "Number of trainable paramters in image_encoder:  {:.5f}M".format(sum(p.numel() for p in tpsam_model.image_encoder.parameters() if p.requires_grad)/1e6)
        ) # 25,319,664 25M
    logging.info(
        "Number of trainable paramters in prompt_generator: {:.5f}M".format(sum(p.numel() for p in tpsam_model.prompt_generator.parameters() if p.requires_grad)/1e6)
        ) # 1,114,624 1M
    logging.info(
        "Number of trainable paramters in mask_decoder: : {:.5f}M".format(sum(p.numel() for p in tpsam_model.mask_decoder.parameters() if p.requires_grad)/1e6)
        ) # 4,058,340 4M

def main(args):
    logging.basicConfig(filename=args.log_txt, level=logging.INFO,
						format='%(asctime)s - %(levelname)s -%(message)s')
    
    logging.info("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device) if torch.cuda.is_available() else 'cpu'

    model_save_path = args.model_save_path

    # load data
    train_dataset = NpyDataset(
		image_paths = args.image_paths,
		mask_paths = args.mask_paths,
		prior_paths = args.prior_paths,
        mode='train'
	)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)

    valid_dataset = NpyDataset(
		image_paths = args.image_paths,
		mask_paths = args.mask_paths,
		prior_paths = args.prior_paths,
        mode='valid'
	)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    test_dataset = NpyDataset(
		image_paths = args.image_paths,
		mask_paths = args.mask_paths,
		prior_paths = args.prior_paths,
        mode='test'
	)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    logging.info("========= Dataset Loading Done ===============")

    # load model
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    prompt_generator = PromptGenerator(num_classes=1, dense_channels=256, sparse_channels=256).to(device)
    tpsam_model = TPSAM(sam_model=sam_model, prompt_generator=prompt_generator).to(device)
    logging_model_info(tpsam_model)

    logging.info("========= Model Init Done ===============")
    tpsam_model.train()

    # load optimizer
    optimizer = torch.optim.AdamW(
        tpsam_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # load loss
    dice_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    losses_dice = []
    losses_ce = [] 
    losses = []
    best_loss = 1e10

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"]+1
            tpsam_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
    if args.use_amp:
        scaler=torch.cuda.amp.GradScaler()
    
    writer = SummaryWriter(log_dir=args.tsbd_log)

    for epoch in range(start_epoch, args.num_epochs):
        epoch_dice_loss = 0
        epoch_ce_loss = 0
        epoch_loss = 0

        logging.info("Epoch {}:".format(epoch))
        train_labels = torch.tensor([],device=device)
        train_preds = torch.tensor([],device=device)
        
        for step, (image, mask, prior, _) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            image, mask, prior = image.to(device), mask.to(device), prior.to(device)

            if args.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    tpsam_pred = tpsam_model(image, prior)
                    tpsam_pred = tpsam_pred.squeeze(1)

                    loss_dice = dice_loss(tpsam_pred, mask)
                    loss_ce = ce_loss(tpsam_pred, mask.float())
                    loss = loss_dice + loss_ce
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                tpsam_pred = tpsam_model(image, prior)
                tpsam_pred = tpsam_pred.squeeze(1)

                loss_dice = dice_loss(tpsam_pred, mask)
                loss_ce = ce_loss(tpsam_pred, mask.float())
                loss = loss_dice + loss_ce

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            epoch_dice_loss += loss_dice.item()
            epoch_ce_loss += loss_ce.item()
            epoch_loss += loss.item()

            train_labels = torch.cat((train_labels, mask), dim=0)
            train_preds = torch.cat((train_preds, tpsam_pred),dim=0)
            
        epoch_dice_loss /= step
        epoch_ce_loss /= step
        epoch_loss /= step

        losses_dice.append(epoch_dice_loss)
        losses_ce.append(epoch_ce_loss)
        losses.append(epoch_loss)
        
        train_labels = train_labels.detach().cpu().numpy()
        train_preds = train_preds.detach().cpu().numpy()
        train_preds = (train_preds>=0.5).astype(int)
        
        train_IoU = compute_binary_iou(train_preds, train_labels)
        train_DSC = compute_dice_coefficient(train_preds, train_labels)
        train_REC = compute_binary_recall(train_preds, train_labels)

        print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M%S")}, Epoch: {epoch}, Loss: {epoch_loss}')

        logging.info("Epoch {} Train! train_Total_Loss: {:.4f}, train_DICE_Loss: {:.4f}, train_BCE _Loss: {:.4f}".format(epoch, epoch_loss, epoch_dice_loss, epoch_ce_loss))
        logging.info("Epoch {} Train! train_IoU: {:.4f}, train_DSC: {:.4f}, train_REC _Loss: {:.4f}".format(epoch, train_IoU, train_DSC, train_REC))

        writer.add_scalar('Loss/Train Total Loss', epoch_loss, epoch)
        writer.add_scalar('Loss/Train Dice Loss', epoch_dice_loss, epoch)
        writer.add_scalar('Loss/Train BCE Loss', epoch_ce_loss, epoch)

        writer.add_scalar('Metrics/train_IoU', train_IoU, epoch)
        writer.add_scalar('Metrics/train_DSC', train_DSC, epoch)
        writer.add_scalar('Metrics/train_REC', train_REC, epoch)
        
        ## save the latest model
        checkpoint = {
            "model": tpsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, os.path.join(model_save_path, "tpsam_model_latest.pth"))
        logging.info("Epoch {} the Latest Model!".format(epoch))

        ## save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = {
            "model": tpsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            }
            torch.save(checkpoint, os.path.join(model_save_path, "tpsam_model_best.pth"))
            logging.info("Epoch {} New Best Model!".format(epoch))

        ## plot loss
        plt.plot(losses)
        plt.title("Dice + Cross Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(args.logging_dir, "train_loss.png"))
        plt.close()

        # Test loop
        tpsam_model.eval()

        valid_labels = torch.tensor([],device=device)
        valid_preds = torch.tensor([],device=device)
        with torch.no_grad():
            for step, (image, mask, prior, _) in enumerate(tqdm(valid_dataloader)):
                image, mask, prior = image.to(device), mask.to(device), prior.to(device)

                tpsam_pred = tpsam_model(image, prior)
                tpsam_pred = tpsam_pred.squeeze(1)

                # Collect predictions and labels
                valid_labels = torch.cat((valid_labels, mask), dim=0)
                valid_preds = torch.cat((valid_preds, tpsam_pred), dim=0)

            valid_labels = valid_labels.detach().cpu().numpy()
            valid_preds = valid_preds.detach().cpu().numpy()
            valid_preds = (valid_preds >= 0.5).astype(int)
            
            valid_IoU = compute_binary_iou(valid_preds, valid_labels)
            valid_DSC = compute_dice_coefficient(valid_preds, valid_labels)
            valid_REC = compute_binary_recall(valid_preds, valid_labels)

            logging.info("Valid! valid_IoU: {:.4f}, valid_DSC: {:.4f}, valid_REC: {:.4f}".format(valid_IoU, valid_DSC, valid_REC))

            writer.add_scalar('Metrics/valid_IoU', valid_IoU, epoch)
            writer.add_scalar('Metrics/valid_DSC', valid_DSC, epoch)
            writer.add_scalar('Metrics/valid_REC', valid_REC, epoch)

        test_labels = torch.tensor([],device=device)
        test_preds = torch.tensor([],device=device)
        with torch.no_grad():
            for step, (image, mask, prior, _) in enumerate(tqdm(test_dataloader)):
                image, mask, prior = image.to(device), mask.to(device), prior.to(device)

                tpsam_pred = tpsam_model(image, prior)
                tpsam_pred = tpsam_pred.squeeze(1)

                # Collect predictions and labels
                test_labels = torch.cat((test_labels, mask), dim=0)
                test_preds = torch.cat((test_preds, tpsam_pred), dim=0)

            test_labels = test_labels.detach().cpu().numpy()
            test_preds = test_preds.detach().cpu().numpy()
            test_preds = (test_preds >= 0.5).astype(int)
            
            test_IoU = compute_binary_iou(test_preds, test_labels)
            test_DSC = compute_dice_coefficient(test_preds, test_labels)
            test_REC = compute_binary_recall(test_preds, test_labels)

            logging.info("Test! test_IoU: {:.4f}, test_DSC: {:.4f}, test_REC: {:.4f}".format(test_IoU, test_DSC, test_REC))

            writer.add_scalar('Metrics/test_IoU', test_IoU, epoch)
            writer.add_scalar('Metrics/test_DSC', test_DSC, epoch)
            writer.add_scalar('Metrics/test_REC', test_REC, epoch)

    writer.close()

def test():
    vit_b_checkpoint = "/data/home/litingyao/project/SAM/LearnablePromptSAM/sam_checkpoints/pesam_vit_b_01ec64.pth"
    sam_model = sam_model_registry['vit_b'](checkpoint=vit_b_checkpoint)

    device="cuda:0" if torch.cuda.is_available() else "cpu"
    prompt_generator = PromptGenerator(num_classes=1, dense_channels=256, sparse_channels=256).to(device)
    tpsam_model = TPSAM(sam_model=sam_model, prompt_generator=prompt_generator).to(device)

    idrid_dataset = MyDataset(
            image_paths = '/data/home/litingyao/project/SAM/data/IDRiD_NPY/Images/a. Training Set',
            mask_paths = '/data/home/litingyao/project/SAM/data/IDRiD_NPY/Masks/a. Training Set/3. Hard Exudates',
            prior_paths = '/data/home/litingyao/project/SAM/data/IDRiD_NPY/Prior/a. Training Set/3. Hard Exudates'
        )
    idrid_dataloader = DataLoader(idrid_dataset, batch_size=2, num_workers=4, shuffle=True)
    optimizer = torch.optim.AdamW(
        tpsam_model.parameters(), lr=0.001, weight_decay=1e-6
    )

    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    epoch_loss = 0
    for step, (image, mask, prior, _) in enumerate(tqdm(idrid_dataloader)):
        optimizer.zero_grad()
        image, mask, prior = image.to(device), mask.to(device), prior.to(device)
        tpsam_pred = tpsam_model(image, prior)
        print('step:===>', step)
        print('tpsam_pred:==>', tpsam_pred.shape)
        tpsam_pred = tpsam_pred.squeeze(1)
        print('tpsam_pred:==>', tpsam_pred.shape)
        print('mask:==>', mask.shape)
        loss = seg_loss(tpsam_pred, mask) + ce_loss(tpsam_pred, mask.float())
        print('loss:==>', loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    epoch_loss += loss.item()


if __name__ == '__main__':
    args = parse_argus()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    _seed_torch(args)
    main(args)