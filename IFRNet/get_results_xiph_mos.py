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
parser.add_argument('--data_root', type=str, default='/home/abhishri/realtime_vfi/sample_mos_data/xiph', required=True)
parser.add_argument('--model_path', type=str, default='./ifrnet_model/IFRNet/IFRNet_Vimeo90K.pth', required=True)
args = parser.parse_args()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print("Running on", str(device).upper())

model = Model().to(device).eval()
check_point = torch.load(args.model_path)

model.load_state_dict(check_point)
print("...Model Loaded...")

video_filename = os.listdir(args.data_root)
out_dir = 'xiph_ifrnet_' + str(device)
if os.path.exists(out_dir) == False:
    os.makedirs(out_dir)

psnr_list = []
ssim_list = []
infer_time_list = []
filename_list = []

for strFile in video_filename:
    print("Currently Processing: ", strFile)
    for intFrame in range(2, 99, 2):
        # print(data_path + '/' + strFile + '/' + strFile + '-' + str(intFrame - 1).zfill(3) + '.png')
        if os.path.exists(args.data_root + '/' + strFile + '/' + strFile + '-' + str(intFrame).zfill(3) + '.png') == True:
            img0_np = read(args.data_root + '/' + strFile + '/' + strFile + '-' + str(intFrame - 1).zfill(3) + '.png')
            img1_np = read(args.data_root + '/' + strFile + '/' + strFile + '-' + str(intFrame + 1).zfill(3) + '.png')
            gt_np = read(args.data_root + '/' + strFile + '/' + strFile + '-' + str(intFrame).zfill(3) + '.png')

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
            filename_list.append(strFile + '-' + str(intFrame))

        # imgt_pred_np = cv2.cvtColor(imgt_pred_np, cv2.COLOR_BGR2RGB)
        # img0_np = cv2.cvtColor(img0_np, cv2.COLOR_BGR2RGB)
        # img1_np = cv2.cvtColor(img1_np, cv2.COLOR_BGR2RGB)

        # img0_path = os.path.join(out_path, strFile + '-' + str(intFrame - 1).zfill(3) + '.png')
        # out_img_path = os.path.join(out_path, strFile + '-' + str(intFrame).zfill(3) + '.png') 
        # img1_path = os.path.join(out_path, strFile + '-' + str(intFrame + 1).zfill(3) + '.png')

        # cv2.imwrite(out_img_path, imgt_pred_np)
        # cv2.imwrite(img0_path, img0_np)
        # cv2.imwrite(img1_path, img1_np)

video_excel = pd.DataFrame({'Filename':filename_list,'PSNR': psnr_list, 'SSIM': ssim_list, 'Inference Time(s)': infer_time_list})
video_excel.to_excel(out_dir + "/Evaluation.xlsx")
print("Done")
