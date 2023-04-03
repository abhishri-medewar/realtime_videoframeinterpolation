import os
from eval import interpolator, util
from losses import losses
import argparse
import numpy as np
import tensorflow as tf
import pandas as pd
import time

#available models:
#1. /home/abhishri/realtime_vfi/Film/training_film_Style_steps156000/vimeo90K/saved_model
#2. /home/abhishri/realtime_vfi/Film/training_film_L1_lite_steps2148000/vimeo90K/saved_model
#3. /home/abhishri/realtime_vfi/Film/pretrained_models/film_net/L1/saved_model
#4. /home/abhishri/realtime_vfi/Film/pretrained_models/film_net/Style/saved_model

os.environ["CUDA_VISIBLE_DEVICES"] = ""

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('-input_data_path', type=str, default='/home/abhishri/realtime_vfi/mos_data_vimeo90k/sequences/')
parser.add_argument('-model_path', type=str, 
default='/home/abhishri/realtime_vfi/Film/training_film_L1_lite_steps2148000/vimeo90K/saved_model')
parser.add_argument('-output_path', type=str, default='film_L1_s2148000_vimeo90k_cpu')

args = parser.parse_args()

interpolator = interpolator.Interpolator(args.model_path, None)
#Batched time.
batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)


dir_list = ['00001/0389', '00001/0505', '00001/0828', '00002/0843', '00006/0511', '00007/0004',
            '00007/0620', '00007/0854', '00013/0232', '00013/0289', '00013/0449', '00024/0731', 
            '00029/0069', '00029/0795', '00029/0801']


psnr_list = []
ssim_list = []
filename_list = []
infer_time_list = []

if os.path.exists(args.output_path) == False:
    os.makedirs(args.output_path)

for data in dir_list:

    # FIrst batched image.
    image_1 = util.read_image(args.input_data_path + data + '/im1.png')
    image_batch_1 = np.expand_dims(image_1, axis=0)

    # Second batched image.
    image_2 = util.read_image(args.input_data_path + data + '/im3.png')
    image_batch_2 = np.expand_dims(image_2, axis=0)

    #ground truth image
    gt_img = util.read_image(args.input_data_path + data + '/im2.png')
    gt_np = np.expand_dims(gt_img, axis=0)

    # Invoke the model once.
    time_start = time.time()
    mid_frame = interpolator.interpolate(image_batch_1, image_batch_2, batch_dt)[0]
    time_end = time.time()
    # name = data.replace('/', '_')
    # img1_path = os.path.join(args.output_path, name, 'im1.png')
    # out_img_path = os.path.join(args.output_path, name, 'im2.png') 
    # img2_path = os.path.join(args.output_path, name, 'im3.png')

    # util.write_image(img1_path, image_1)
    # util.write_image(out_img_path, mid_frame)
    # util.write_image(img2_path, image_2)
    ssim = tf.reduce_mean(tf.image.ssim(mid_frame, gt_np, max_val=1.0))
    psnr = tf.reduce_mean(tf.image.psnr(mid_frame, gt_np, max_val=1.0))
    inference_time = time_end - time_start
    filename_list.append(data)
    psnr_list.append(psnr.numpy())
    ssim_list.append(ssim.numpy())
    infer_time_list.append(inference_time)

excel_data = pd.DataFrame({'Filename':filename_list,'PSNR': psnr_list, 'SSIM': ssim_list, 'Inference Time(s)': infer_time_list})
excel_data.to_excel(args.output_path + "/Film_vimeo90k_Evaluation.xlsx")
print("...Vimeo90k evaluation complete...")


