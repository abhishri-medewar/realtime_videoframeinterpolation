import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

class XiphTriplet(Dataset):
    def __init__(self, data_root):
        
        self.testlist = os.listdir(data_root)
        
    def __getitem__(self, index):

        imgpath = os.path.join(self.image_root, self.testlist[index])
        imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png']

        # Load images
        img1 = Image.open(imgpaths[0])
        img2 = Image.open(imgpaths[1])
        img3 = Image.open(imgpaths[2])

        T = transforms.ToTensor()
        img1 = T(img1)
        img2 = T(img2)
        img3 = T(img3)

        imgs = [img1, img2, img3]
        
        return imgs, imgpaths

    def __len__(self):

        return len(self.testlist)



def get_loader(mode, data_root, batch_size=1, shuffle=False, num_workers=0, test_mode=None):

    dataset = XiphTriplet(data_root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
