import os
import sys
sys.path.append('.')
import torch
import numpy as np
from utils_misc import read
from metric import calculate_psnr, calculate_ssim
from models.IFRNet import Model
import cv2
from tqdm import tqdm
import time
import pandas as pd
from collections import OrderedDict

# from models.IFRNet_L import Model
# from models.IFRNet_S import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Model()

check_point = torch.load(r'/IFRNetmain/checkpoint/IFRNet/2022-07-18 20:30:46/IFRNet_best.pth')
# check_point = torch.load(r'/IFRNetmain/checkpoint/IFRNet/2022-07-18 20:30:46/IFRNet_latest.pth')

new_state_dict = OrderedDict()
for k, v in check_point.items():
    name = k[7:] # remove 'module.' of dataparallel
    new_state_dict[name]=v

model.load_state_dict(new_state_dict)

#this is the model provided in paper
# model.load_state_dict(torch.load('/IFRNetmain/IFRNet_Vimeo90K.pth'))  

model.eval()
model.cuda()

# Replace the 'path' with your UCF101 dataset absolute path.
path = '/IFRNetmain/ucf101_interp_ours'
dirs = sorted(os.listdir(path))

psnr_list = []
ssim_list = []
folder_no = []
infer_time = []

for d in tqdm(dirs):
    I0 = read(path + '/' + d + '/frame_00.png')
    I1 = read(path + '/' + d + '/frame_01_gt.png')
    I2 = read(path + '/' + d + '/frame_02.png')

    I0 = (torch.tensor(I0.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
    I1 = (torch.tensor(I1.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
    I2 = (torch.tensor(I2.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
    embt = torch.tensor(1/2).float().view(1, 1, 1, 1).to(device)

    time_start = time.time()
    I1_pred = model.inference(I0, I2, embt)
    time_end = time.time()

    I1_pred_orig = (I1_pred[0].data.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    path_out = os.path.join('./output/best', d + '_' + 'out.png')
    # path = os.path.join('./output/latest', d + '_' + 'out.png')
    # path = os.path.join('./output/best', d + '_' + 'out.png')
    cv2.imwrite(path_out, I1_pred_orig)

    psnr = calculate_psnr(I1_pred, I1).detach().cpu().numpy()
    ssim = calculate_ssim(I1_pred, I1).detach().cpu().numpy()

    # psnr_list.append(psnr)
    # ssim_list.append(ssim)
    folder_no.append(d)
    psnr_list.append(psnr)
    ssim_list.append(ssim)
    infer_time.append(time_end - time_start)
    
    print("PSNR: ", psnr)
    print("SSIM: ", ssim)

    # print(('Avg PSNR: {} SSIM: {}'.format(np.mean(psnr_list), np.mean(ssim_list))))
    #write metric evaluation for ucf101 to excel
    # ucf_evaluation = pd.DataFrame({'Directory No:':folder_no, 'PSNR': psnr_list, 'SSIM': ssim_list, 'Inference Time(s)': infer_time})
    # ucf_evaluation.to_excel("./output/UCFEval_latest.xlsx")
    # ucf_evaluation.to_excel("./output/UCFEval_latest.xlsx")
    # ucf_evaluation.to_excel("./output/UCFEval_best.xlsx")
