import os
import pandas as pd
from eval import interpolator as interpolator_lib
from eval import util
from losses import losses
import argparse
import numpy as np
import tensorflow as tf
from typing import Generator, Iterable, List, Optional
import time

#available models:
#1. /home/abhishri/realtime_vfi/Film/training_film_Style_steps156000/vimeo90K/saved_model
#2. /home/abhishri/realtime_vfi/Film/training_film_L1_lite_steps2148000/vimeo90K/saved_model
#3. /home/abhishri/realtime_vfi/Film/pretrained_models/film_net/L1/saved_model
#4. /home/abhishri/realtime_vfi/Film/pretrained_models/film_net/Style/saved_model

os.environ["CUDA_VISIBLE_DEVICES"] = ""

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('-model_path', type=str, default='/home/abhishri/realtime_vfi/Film/training_film_L1_lite_steps2148000/vimeo90K/saved_model')
parser.add_argument('-output_path', type=str, default='film_L1_s2148000_xiph_cpu')
parser.add_argument('-input_data_path', type=str, default='/home/abhishri/realtime_vfi/sample_mos_data/xiph')

args = parser.parse_args()

interpolator = interpolator_lib.Interpolator(args.model_path, None)
#Batched time.
batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)

if os.path.exists(args.output_path) == False:
    os.makedirs(args.output_path)
video_filename = os.listdir(args.input_data_path)

infer_time_list = []
psnr_list = []
ssim_list = []
filename_list = []

for strFile in video_filename:
    print("Currently Processing: ", strFile)
    for intFrame in range(2, 99, 2):

            if os.path.exists(args.input_data_path + '/' + strFile + '/' + strFile + '-' + str(intFrame).zfill(3) + '.png'):

                # First batched image.
                image_1 = util.read_image(args.input_data_path + '/' + strFile + '/' + strFile + '-' + str(intFrame - 1).zfill(3) + '.png')
                image_batch_1 = np.expand_dims(image_1, axis=0)

                # Second batched image.
                image_2 = util.read_image(args.input_data_path + '/' + strFile + '/' + strFile + '-' + str(intFrame + 1).zfill(3) + '.png')
                image_batch_2 = np.expand_dims(image_2, axis=0)

                #ground truth image
                gt_img = util.read_image(args.input_data_path + '/' + strFile + '/' + strFile + '-' + str(intFrame).zfill(3) + '.png')
                gt_np = np.expand_dims(gt_img, axis=0)

                # Invoke the model once.
                time_start = time.time()
                mid_frame = interpolator.interpolate(image_batch_1, image_batch_2, batch_dt)[0]
                time_end = time.time()
                # img1_path = os.path.join(out_path, strFile + '-' + str(intFrame - 1).zfill(3) + '.png')
                # out_img_path = os.path.join(out_path, strFile + '-' + str(intFrame).zfill(3) + '.png') 
                # img2_path = os.path.join(out_path, strFile + '-' + str(intFrame + 1).zfill(3) + '.png')

                # util.write_image(img1_path, image_1)
                # util.write_image(out_img_path, mid_frame)
                # util.write_image(img2_path, image_2)
                ssim = tf.reduce_mean(tf.image.ssim(mid_frame, gt_np, max_val=1.0))
                psnr = tf.reduce_mean(tf.image.psnr(mid_frame, gt_np, max_val=1.0))
                inference_time = time_end - time_start
                infer_time_list.append(inference_time)
                filename_list.append(strFile + '-' + str(intFrame))
                psnr_list.append(psnr.numpy())
                ssim_list.append(ssim.numpy())

    
excel_data = pd.DataFrame({'Filename':filename_list,'PSNR': psnr_list, 'SSIM': ssim_list, 'Inference Time(s)': infer_time_list})
excel_data.to_excel(args.output_path + "/Film_xiph_Evaluation.xlsx")
print("...Xiph evaluation complete...")

