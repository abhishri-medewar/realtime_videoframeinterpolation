import os
import cv2
import torch
import argparse
from torch.nn import functional as F
import warnings
import glob
from imageio import mimsave
import time
from tqdm import tqdm
import pandas as pd
from model.pytorch_msssim import ssim_matlab
import numpy as np
import math

warnings.filterwarnings("ignore")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')

print("Running on", str(device).upper())

torch.set_grad_enabled(False)
if device == 'cuda':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--img', dest='img', nargs=2)
parser.add_argument('--exp', default=1, type=int)
parser.add_argument('--ratio', default=0, type=float, help='inference ratio between two images with 0 - 1 range')
parser.add_argument('--rthreshold', default=0.02, type=float, help='returns image when actual ratio falls in given range threshold')
parser.add_argument('--rmaxcycles', default=8, type=int, help='limit max number of bisectional cycles')
parser.add_argument('--model', dest='modelDir', type=str, default='rife_model/RIFE/train_log', help='directory with trained model files')
parser.add_argument('--data_root',, type=str, default='/home/abhishri/realtime_vfi/mos_data_vimeo90k/sequences/')
args = parser.parse_args()

from model.RIFE import Model
model = Model()
model.load_model(args.modelDir, -1)
print("Loaded ArXiv-RIFE model")

model.eval()
model.device()

dir_list = ['00001/0389', '00001/0505', '00001/0828', '00002/0843', '00006/0511', '00007/0004',
'00007/0620' ,'00007/0854', '00013/0232', '00013/0289', '00013/0449', '00024/0731', 
'00029/0069', '00029/0795', '00029/0801']

out_dir = 'vimeo90k_rife_' + str(device)
if os.path.exists(out_dir) == False:
    os.makedirs(out_dir)

file_name = []
psnr_list = []
ssim_list = []
infer_time_list = []
filename_list = []

for strFile in dir_list:
    print("Currently Processing: ", strFile)
    img0 = cv2.imread(args.data_root + strFile + '/im1.png', cv2.IMREAD_UNCHANGED)
    img1 = cv2.imread(args.data_root + strFile + '/im3.png', cv2.IMREAD_UNCHANGED)
    gt = cv2.imread(args.data_root + strFile + '/im2.png', cv2.IMREAD_UNCHANGED)

    img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    gt = (torch.tensor(gt.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

    n, c, h, w = img0.shape

    with torch.no_grad():
        time_start = time.time()
        mid = model.inference(img0, img1)[0]
        time_end = time.time()
    
    ssim = ssim_matlab(gt, torch.round(mid * 255).unsqueeze(0) / 255.).detach().cpu().numpy()
    out = mid.detach().cpu().numpy().transpose(1, 2, 0)
    out = np.round(out * 255) / 255.
    gt = gt[0].cpu().numpy().transpose(1, 2, 0)
    psnr = -10 * math.log10(((gt - out) * (gt - out)).mean())
    inference_time = time_end - time_start

    psnr_list.append(psnr)
    ssim_list.append(ssim)
    infer_time_list.append(inference_time)
    filename_list.append(strFile)

    # name = data.replace('/', '_')
    # img0_path = os.path.join('/VFcomparison/RIFE/vimeo90k_out/', name, 'im1.png')
    # out_img_path = os.path.join('/VFcomparison/RIFE/vimeo90k_out/', name, 'im2_out.png') 
    # img1_path = os.path.join('/VFcomparison/RIFE/vimeo90k_out/', name, 'im3.png')

    # cv2.imwrite(img0_path, (img0[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
    # cv2.imwrite(out_img_path, (mid * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
    # cv2.imwrite(img1_path, (img1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])

video_excel = pd.DataFrame({'Filename':filename_list,'PSNR': psnr_list, 'SSIM': ssim_list, 'Inference Time(s)': infer_time_list})
video_excel.to_excel(out_dir + "/Evaluation.xlsx")
print("Done")