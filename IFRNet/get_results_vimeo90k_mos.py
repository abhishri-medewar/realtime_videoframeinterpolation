import os
import numpy as np
import torch
from models.IFRNet import Model
from utils_misc import read
from imageio import mimsave
from collections import OrderedDict
import cv2
import glob
from metric import calculate_psnr, calculate_ssim
import time
from tqdm import tqdm
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='/home/abhishri/realtime_vfi/mos_data_vimeo90k', required=True)
parser.add_argument('--model_path', type=str, default='./ifrnet_model/IFRNet/IFRNet_Vimeo90K.pth', required=True)
args = parser.parse_args()

dir_list = ['00001/0389', '00001/0505', '00001/0828', '00002/0843', '00006/0511', '00007/0004',
'00007/0620' ,'00007/0854', '00013/0232', '00013/0289', '00013/0449', '00024/0731', 
'00029/0069', '00029/0795', '00029/0801']

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
model = Model().to(device).eval()
check_point = torch.load(args.model_path)

model.load_state_dict(check_point)
print("...Model Loaded...")

# data_root = '/home/abhishri/realtime_vfi/mos_data_vimeo90k/sequences/'
out_dir = 'vimeo90k_ifrnet_' + str(device)
if os.path.exists(out_dir) == False:
    os.makedirs(out_dir)

psnr_list = []
ssim_list = []
infer_time_list = []
filename_list = []
for strFile in dir_list:
    print("Currently Processing: ", strFile)
    # name = strFile.replace('/', '_')
    # out_path = os.path.join('/VFcomparison/IFRNet/', out_dir , name)
    # if os.path.exists(out_path) == False:
    #     os.makedirs(out_path)

    img0_np = read(args.data_root + strFile + '/im1.png')
    img1_np = read(args.data_root + strFile + '/im3.png')
    gt_np = read(args.data_root + strFile + '/im2.png')

    img0 = (torch.tensor(img0_np.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
    img1 = (torch.tensor(img1_np.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
    embt = torch.tensor(1/2).view(1, 1, 1, 1).float().to(device)

    gt = (torch.tensor(gt_np.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device) 

    with torch.no_grad():
        time_start = time.time()
        imgt_pred = model.inference(img0, img1, embt)
        time_end = time.time()

    imgt_pred_np = (imgt_pred[0].data.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

    psnr = calculate_psnr(imgt_pred, gt).detach().cpu().numpy()
    ssim = calculate_ssim(imgt_pred, gt).detach().cpu().numpy()
    inference_time = time_end - time_start
    psnr_list.append(psnr)
    ssim_list.append(ssim)
    infer_time_list.append(inference_time)
    filename_list.append(strFile)

    # imgt_pred_np = cv2.cvtColor(imgt_pred_np, cv2.COLOR_BGR2RGB)
    # img0_np = cv2.cvtColor(img0_np, cv2.COLOR_BGR2RGB)
    # img1_np = cv2.cvtColor(img1_np, cv2.COLOR_BGR2RGB)

    # img0_path = os.path.join('/VFcomparison/IFRNet/'+ out_dir + '/' + name + '/im1.png')
    # out_img_path = os.path.join('/VFcomparison/IFRNet/'+ out_dir + '/' + name + '/im2.png') 
    # img1_path = os.path.join('/VFcomparison/IFRNet/'+ out_dir + '/' + name + '/im3.png')

    # cv2.imwrite(out_img_path, imgt_pred_np)
    # cv2.imwrite(img0_path, img0_np)
    # cv2.imwrite(img1_path, img1_np)

video_excel = pd.DataFrame({'Filename':filename_list,'PSNR': psnr_list, 'SSIM': ssim_list, 'Inference Time(s)': infer_time_list})
video_excel.to_excel(out_dir + "/Evaluation.xlsx")
print("Done")