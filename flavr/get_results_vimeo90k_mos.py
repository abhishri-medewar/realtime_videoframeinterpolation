import os
import numpy as np
import torch
from collections import OrderedDict
import cv2
import glob
import time
from tqdm import tqdm
import pandas as pd
from model.FLAVR_arch import UNet_3D_3D
from PIL import Image
from torchvision import transforms
import myutils
from pytorch_msssim import ssim_matlab as calc_ssim
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='/home/abhishri/realtime_vfi/mos_data_vimeo90k/sequences/', required=True)
parser.add_argument('--model_path', type=str, default='/home/abhishri/realtime_vfi/flavr/trained_model/FLAVR_2x.pth', required=True)
args = parser.parse_args()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print("Running using", device)

from model.FLAVR_arch import UNet_3D_3D
model_name = "unet_18"
n_inputs = 4
#for 2x interpolation
n_outputs = 1

model = UNet_3D_3D(model_name, n_inputs=n_inputs, n_outputs=n_outputs)
model = torch.nn.DataParallel(model)
checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint["state_dict"] , strict=True)
del checkpoint
model.to(device)

#remove the DataParallel wrapper
if device == torch.device('cpu'):
    model = model.module
model.eval()

print("...Model Loaded...")
print("#params" , sum([p.numel() for p in model.parameters()]))


dir_list = ['00001/0389', '00001/0505', '00001/0828', '00002/0843', '00006/0511', '00007/0004',
'00007/0620' ,'00007/0854', '00013/0232', '00013/0289', '00013/0449', '00024/0731', 
'00029/0069', '00029/0795', '00029/0801']

# data_root = '/home/abhishri/realtime_vfi/mos_data_vimeo90k/sequences/'
out_dir = 'flavr_vimeo90k_mos_' + str(device)
if os.path.exists(out_dir) == False:
    os.makedirs(out_dir)

psnr_list = []
ssim_list = []
infer_time_list = []
filename_list = []

for strFile in dir_list:
    print("Currently Processing: ", strFile)

    img0 = Image.open(args.data_root + strFile + '/im1.png')
    img1 = Image.open(args.data_root + strFile + '/im3.png')
    gt = Image.open(args.data_root + strFile + '/im2.png')

    T = transforms.ToTensor()
    img0 = T(img0)
    img1 = T(img1)
    gt = T(gt)

    img0 = img0.to(device).unsqueeze(0)
    img1 = img1.to(device).unsqueeze(0)
    gt = gt.to(device).unsqueeze(0)

    images = [img0, img0, img1, img1]

    with torch.no_grad():
        time_start = time.time()
        mid = model(images)
        time_end = time.time()
    
    psnr = myutils.calc_psnr(mid[0], gt)
    ssim = calc_ssim(mid[0].clamp(0,1), gt.clamp(0,1) , val_range=1.)
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