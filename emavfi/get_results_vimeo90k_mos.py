import cv2
import math
import sys
import torch
import numpy as np
import argparse
import os
import warnings
import time
from tqdm import tqdm
import pandas as pd
# import torchsummary
warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)

device = torch.device('cpu')

'''==========import from our code=========='''
sys.path.append('.')
import config as cfg
from Trainer import Model
from benchmark.utils.pytorch_msssim import ssim_matlab


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ours_small', type=str)
parser.add_argument('--data_root', type=str, default='/home/abhishri/realtime_vfi/mos_data_vimeo90k/sequences/', required=True)
args = parser.parse_args()
assert args.model in ['ours', 'ours_small'], 'Model not exists!'

'''==========Model setting=========='''
TTA = True
if args.model == 'ours_small':
    TTA = False
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 16,
        depth = [2, 2, 2, 2, 2]
    )
else:
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 32,
        depth = [2, 2, 2, 4, 4]
    )
model = Model(-1)

# model.load_state_dict(torch.load(f'ckpt/ours.pkl', map_location=torch.device('cpu')))
# print("#params" , sum([p.numel() for p in model.parameters()]))
# exit()

model.load_model(rank = -2)
# model.load_model()
model.eval()
# model.to(device)

dir_list = ['00001/0389', '00001/0505', '00001/0828', '00002/0843', '00006/0511', '00007/0004',
'00007/0620' ,'00007/0854', '00013/0232', '00013/0289', '00013/0449', '00024/0731', 
'00029/0069', '00029/0795', '00029/0801']

out_dir = 'emavfi_vimeo90k_' + args.model
if os.path.exists(out_dir) == False:
    os.makedirs(out_dir)

psnr_list = []
ssim_list = []
infer_time_list = []
filename_list = []

for strFile in dir_list:
    print("Currently Processing: ", strFile)

    img0_np = cv2.imread(args.data_root + strFile + '/im1.png')
    img1_np = cv2.imread(args.data_root + strFile + '/im3.png')
    gt_np = cv2.imread(args.data_root + strFile + '/im2.png')

    img0 = (torch.tensor(img0_np.transpose(2, 0, 1)).cpu() / 255.).unsqueeze(0)
    img1 = (torch.tensor(img1_np.transpose(2, 0, 1)).cpu() / 255.).unsqueeze(0)

    with torch.no_grad():
        time_start = time.time()
        mid = model.inference(img0, img1, TTA=TTA, fast_TTA=TTA)[0]
        time_end = time.time()

    ssim = ssim_matlab(torch.tensor(gt_np.transpose(2, 0, 1)).cpu().unsqueeze(0) / 255., mid.unsqueeze(0)).detach().cpu().numpy()
    mid = mid.detach().cpu().numpy().transpose(1, 2, 0) 
    gt_np = gt_np / 255.
    psnr = -10 * math.log10(((gt_np - mid) * (gt_np - mid)).mean())
    inference_time = time_end - time_start
    psnr_list.append(psnr)
    ssim_list.append(ssim)
    infer_time_list.append(inference_time)
    filename_list.append(strFile)

video_excel = pd.DataFrame({'Filename':filename_list,'PSNR': psnr_list, 'SSIM': ssim_list, 'Inference Time(s)': infer_time_list})
video_excel.to_excel(out_dir + "/Evaluation.xlsx")
print("Done")



