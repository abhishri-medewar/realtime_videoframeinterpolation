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
parser.add_argument('--data_root', type=str, default='/home/abhishri/realtime_vfi/mos_data_vimeo90k', required=True)
parser.add_argument('--model_path', type=str, default='pretrained_cain.pth', required=True)
args = parser.parse_args()

device = torch.device('cpu')

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

out_dir = 'cain_vimeo90k_' + str(device)
if os.path.exists(out_dir) == False:
    os.makedirs(out_dir)

psnr_list = []
ssim_list = []
infer_time_list = []
filename_list = []

test_loader = get_loader('test',args.data_root, 1)
for i, (images, meta) in enumerate(tqdm(test_loader)):
    strFile = '/'.join(meta[1][0].split('/')[-3:-1])
    print("Currently Processing: ", strFile) 

    im1, gt, im3 = images[0].to(device), images[1].to(device), images[2].to(device)

    with torch.no_grad():
        time_start = time.time()
        mid, _ = model(im1, im3)
        time_end = time.time()

    #quantize the prediction and the gt
    q_mid = mid.mul(255 / 1.).clamp(0, 255).round()
    q_gt = gt.mul(255 / 1.).clamp(0, 255).round()

    psnr = utils.calc_psnr(q_mid, q_gt, mask=None)
    ssim = ssim_pth(q_mid, q_gt, val_range=255)
    if device == torch.device('cuda'):
        ssim = ssim.cpu()
    inference_time = time_end - time_start
    psnr_list.append(psnr)
    ssim_list.append(ssim.numpy())
    infer_time_list.append(inference_time)
    filename_list.append(strFile)

video_excel = pd.DataFrame({'Filename':filename_list,'PSNR': psnr_list, 'SSIM': ssim_list, 'Inference Time(s)': infer_time_list})
video_excel.to_excel(out_dir + "/Evaluation.xlsx")
print("Done")