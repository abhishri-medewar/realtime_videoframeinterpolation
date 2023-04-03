import os
import sys
import time
import copy
import shutil
import random
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from pytorch_msssim import ssim_matlab as ssim_pth
import config
import utils
from model.cain import CAIN
from torchvision import transforms
from data.vimeo90k import get_loader
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='/home/abhishri/realtime_vfi/sample_mos_data/xiph', required=True)
parser.add_argument('--model_path', type=str, default='pretrained_cain.pth', required=True)
args = parser.parse_args()

device = torch.device('cuda')
# device = torch.device('cpu')

print("Building model: CAIN")
model = CAIN(depth=3)
model = torch.nn.DataParallel(model)
checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['state_dict'])
del checkpoint
model.to(device)

#remove the DataParallel wrapper
if device == torch.device('cpu'):
    model = model.module
model.eval()

video_filename = os.listdir(args.data_root)
out_dir = 'cain_xiph_' + str(device)
if os.path.exists(out_dir) == False:
    os.makedirs(out_dir)

psnr_list = []
ssim_list = []
infer_time_list = []
filename_list = []

for strFile in video_filename:
    print("Currently Processing: ", strFile)
    for intFrame in range(2, 99, 2):
        # print(args.data_root + '/' + strFile + '/' + strFile + '-' + str(intFrame - 1).zfill(3) + '.png')
        if os.path.exists(args.data_root + '/' + strFile + '/' + strFile + '-' + str(intFrame).zfill(3) + '.png') == True:

            img0 = Image.open(args.data_root + '/' + strFile + '/' + strFile + '-' + str(intFrame - 1).zfill(3) + '.png')
            img1 = Image.open(args.data_root + '/' + strFile + '/' + strFile + '-' + str(intFrame + 1).zfill(3) + '.png')
            gt = Image.open(args.data_root + '/' + strFile + '/' + strFile + '-' + str(intFrame).zfill(3) + '.png')

            T = transforms.ToTensor()
            img0 = T(img0)
            img1 = T(img1)
            gt = T(gt)    

            img0, img1, gt = img0.to(device).unsqueeze(0), img1.to(device).unsqueeze(0), gt.to(device).unsqueeze(0)

            with torch.no_grad():
                time_start = time.time()
                mid, _ = model(img0, img1)
                time_end = time.time()

            #quantize the prediction and the gt
            q_mid = mid.mul(255 / 1.).clamp(0, 255).round()
            q_gt = gt.mul(255 / 1.).clamp(0, 255).round()

            # print(q_mid.shape)
            # print(q_gt.shape)
            # exit()

            psnr = utils.calc_psnr(q_mid, q_gt, mask=None)
            ssim = ssim_pth(q_mid, q_gt, val_range=255)
            if device == torch.device('cuda'):
                ssim = ssim.cpu()
            inference_time = time_end - time_start
            psnr_list.append(psnr)
            ssim_list.append(ssim.numpy())
            infer_time_list.append(inference_time)
            filename_list.append(strFile + '-' + str(intFrame))

video_excel = pd.DataFrame({'Filename':filename_list,'PSNR': psnr_list, 'SSIM': ssim_list, 'Inference Time(s)': infer_time_list})
video_excel.to_excel(out_dir + "/Evaluation.xlsx")
print("Done")
