import os
import numpy as np
import torch
from models.IFRNet import Model
from utils_misc import read
from imageio import mimsave
from collections import OrderedDict

# torch.cuda.empty_cache()
# model = Model().cuda().eval()
model = Model().eval()


check_point = torch.load(r'/IFRNetmain/checkpoint/IFRNet/2022-07-18 20:30:46/IFRNet_best.pth')
# check_point = torch.load(r'/IFRNetmain/checkpoint/IFRNet/2022-07-18 20:30:46/IFRNet_latest.pth')

new_state_dict = OrderedDict()
for k, v in check_point.items():
    name = k[7:] # remove 'module.' of dataparallel
    new_state_dict[name]=v

model.load_state_dict(new_state_dict)

# model.load_state_dict(torch.load('./checkpoints/IFRNet/IFRNet_GoPro.pth'))

# img0_np = read('./figures/frame0_4k.png')
# img8_np = read('./figures/frame8_4k.png')

# img0_np = read('./figures/img0.png')
# img8_np = read('./figures/img1.png')

img0_np = read('./figures/frame0_hd.png')
img8_np = read('./figures/frame8_hd.png')

print(img0_np.shape)
print(img8_np.shape)

# img0 = (torch.tensor(img0_np.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
# img8 = (torch.tensor(img8_np.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()

img0 = (torch.tensor(img0_np.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0)
img8 = (torch.tensor(img8_np.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0)

print(img0.shape)

# emb1 = torch.tensor(1/8).view(1, 1, 1, 1).float().cuda()
# emb2 = torch.tensor(2/8).view(1, 1, 1, 1).float().cuda()
# emb3 = torch.tensor(3/8).view(1, 1, 1, 1).float().cuda()
# emb4 = torch.tensor(4/8).view(1, 1, 1, 1).float().cuda()
# emb5 = torch.tensor(5/8).view(1, 1, 1, 1).float().cuda()
# emb6 = torch.tensor(6/8).view(1, 1, 1, 1).float().cuda()
# emb7 = torch.tensor(7/8).view(1, 1, 1, 1).float().cuda()

emb1 = torch.tensor(1/8).view(1, 1, 1, 1).float()
emb2 = torch.tensor(2/8).view(1, 1, 1, 1).float()
emb3 = torch.tensor(3/8).view(1, 1, 1, 1).float()
emb4 = torch.tensor(4/8).view(1, 1, 1, 1).float()
emb5 = torch.tensor(5/8).view(1, 1, 1, 1).float()
emb6 = torch.tensor(6/8).view(1, 1, 1, 1).float()
emb7 = torch.tensor(7/8).view(1, 1, 1, 1).float()

img0_ = torch.cat([img0, img0, img0, img0, img0, img0, img0], 0)
img8_ = torch.cat([img8, img8, img8, img8, img8, img8, img8], 0)
embt = torch.cat([emb1, emb2, emb3, emb4, emb5, emb6, emb7], 0)

print(img0_.shape)
print(img8_.shape)

imgt_pred = model.inference(img0_, img8_, embt)

img1_pred_np = (imgt_pred[0].data.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
img2_pred_np = (imgt_pred[1].data.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
img3_pred_np = (imgt_pred[2].data.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
img4_pred_np = (imgt_pred[3].data.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
img5_pred_np = (imgt_pred[4].data.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
img6_pred_np = (imgt_pred[5].data.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
img7_pred_np = (imgt_pred[6].data.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

images = [img0_np, img1_pred_np, img2_pred_np, img3_pred_np, img4_pred_np, img5_pred_np, img6_pred_np, img7_pred_np, img8_np]
mimsave('./figures/out_8x.gif', images, fps=9)
