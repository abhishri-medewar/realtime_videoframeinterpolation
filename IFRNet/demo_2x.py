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

file_name = [file for file in glob.glob("./figures/video5/*.png")]
file_name.sort(key = lambda x: int(x.split('.')[1].split('/')[-1]))
print("Filenames sorted", file_name[0:5])
model = Model().cuda().eval()

check_point = torch.load(r'/IFRNetmain/checkpoint/IFRNet/2022-07-18 20:30:46/IFRNet_best.pth')
# check_point = torch.load(r'/IFRNetmain/checkpoint/IFRNet/2022-07-18 20:30:46/IFRNet_latest.pth')

new_state_dict = OrderedDict()
for k, v in check_point.items():
    name = k[7:] # remove 'module.' of dataparallel
    new_state_dict[name]=v

model.load_state_dict(new_state_dict)
print("Model Loaded")
# model.load_state_dict(torch.load('./checkpoints/IFRNet/IFRNet_Vimeo90K.pth'))

psnr_list = []
ssim_list = []
infer_time_list = []
filename_list = []

for i in tqdm(range(0, len(file_name), 3)):

    img0_np = read(file_name[i])
    gt_np = read(file_name[i + 1])
    img1_np = read(file_name[i + 2])

    img0 = (torch.tensor(img0_np.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
    img1 = (torch.tensor(img1_np.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
    embt = torch.tensor(1/2).view(1, 1, 1, 1).float().cuda()

    gt = (torch.tensor(gt_np.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda() 

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
    filename_list.append(file_name[i + 1])

    # images = [img0_np, imgt_pred_np, img1_np]
    # mimsave('./figures/out_2x.gif', images, fps=3)
    # images = [imgt_pred_np]
    imgt_pred_np = cv2.cvtColor(imgt_pred_np, cv2.COLOR_BGR2RGB)
    img0_np = cv2.cvtColor(img0_np, cv2.COLOR_BGR2RGB)
    img1_np = cv2.cvtColor(img1_np, cv2.COLOR_BGR2RGB)

    img0_path = os.path.join('./figures/video5_output/', file_name[i].split('.')[1].split('/')[-1] + '.png')
    out_img_path = os.path.join('./figures/video5_output/', file_name[i + 1].split('.')[1].split('/')[-1] + '_out.png') 
    img1_path = os.path.join('./figures/video5_output/', file_name[i + 2].split('.')[1].split('/')[-1] + '.png')

    cv2.imwrite(out_img_path, imgt_pred_np)
    cv2.imwrite(img0_path, img0_np)
    cv2.imwrite(img1_path, img1_np)
    # mimsave('./figures/out_2x.gif', images, fps=1)
video1_excel = pd.DataFrame({'Filename':filename_list,'PSNR': psnr_list, 'SSIM': ssim_list, 'Inference Time(s)': infer_time_list})
video1_excel.to_excel("./figures/Video5_Evaluation.xlsx")







