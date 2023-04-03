import os
import sys
import time
import copy
import shutil
import random

import torch
import numpy as np
from tqdm import tqdm

import config
import utils

from PIL import Image

from model.cain import CAIN
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Building model: CAIN")
model = CAIN(depth=3)
model = torch.load('pretrained_cain.pth')
print("..Completed..")

T = transforms.ToTensor()
img1 = Image.open('./example/img1.jpg')
img2 = Image.open('./example/img2.jpg')
img1 = T(img1)
img2 = T(img2)

out, _ = model(img1, img2)
utils.save_image(out, './out.jpg')