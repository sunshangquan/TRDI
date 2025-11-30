import json
import argparse
import os
import numpy as np
from PIL import Image
import csv
import sys

from matrics_calculator import MetricsCalculator

def mask_decode(encoded_mask,image_shape=[512,512]):
    length=image_shape[0]*image_shape[1]
    mask_array=np.zeros((length,))
    
    for i in range(0,len(encoded_mask),2):
        splice_len=min(encoded_mask[i+1],length-encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i]+j]=1
            
    mask_array=mask_array.reshape(image_shape[0], image_shape[1])
    # to avoid annotation errors in boundary
    mask_array[0,:]=1
    mask_array[-1,:]=1
    mask_array[:,0]=1
    mask_array[:,-1]=1
            
    return mask_array

def align_size(img1, img2):
    if img1.size != img2.size:
        img2 = img2.resize(img1.size)
    return img1, img2

def calculate_metric(metrics_calculator,metric, src_image, tgt_image):
    if metric=="psnr":
        return metrics_calculator.calculate_psnr(src_image, tgt_image, None, None)
    if metric=="lpips":
        return metrics_calculator.calculate_lpips(src_image, tgt_image, None, None)
    if metric=="mse":
        return metrics_calculator.calculate_mse(src_image, tgt_image, None, None)
    if metric=="ssim":
        return metrics_calculator.calculate_ssim(src_image, tgt_image, None, None)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics',  nargs = '+', type=str, default=[
                                                         "psnr",
                                                         "lpips",
                                                         "mse",
                                                         "ssim",
                                                         ])

    parser.add_argument('--tgt_path', type=str, default='/data/COCO_val2017/')
    parser.add_argument('--tgt_methods', type=str,)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--evaluate_whole_table', action= "store_true") # rerun existing images

    args = parser.parse_args()
    
    metrics=args.metrics
    tgt_methods=args.tgt_methods if args.tgt_methods[-1] != '/' else args.tgt_methods[:-1]
    
    metrics_calculator=MetricsCalculator(args.device)
    tgt_result_files = []
    tgt_image_folders = []
    all_results = []
    

    files = os.listdir(args.tgt_methods)
    tgt_result_file = tgt_methods + ".csv"

    for key in files:
        evaluation_result=[key]
        print(f"evaluating image {key} ...")  
        src_image_file = os.path.join(args.tgt_path, key)  
        src_image = Image.open(src_image_file)
        res_image_file = os.path.join(tgt_methods, key)
        

        print(f"evluating method: {res_image_file} {src_image_file}")
        
        tgt_image = Image.open(res_image_file)
        print(src_image.size, tgt_image.size)
        src_image, tgt_image = align_size(src_image, tgt_image)
        print(np.array(src_image).max() , np.array(tgt_image).max(), np.mean((np.array(src_image)/255 - np.array(tgt_image)/255)**2)) 
        for metric in metrics:
            print(f"evluating metric: {metric}")
            evaluation_result.append(calculate_metric(metrics_calculator, metric, src_image, tgt_image))
                    
        with open(tgt_result_file,'a+',newline="") as f:
            csv_write = csv.writer(f)
            csv_write.writerow(evaluation_result)

        all_results.append(evaluation_result[1:])
    
    print(tgt_result_file)
    
    result = np.array(all_results).astype(np.float32)
    print(result)
    with open(tgt_result_file,'a+',newline="") as f:
        csv_write = csv.writer(f)
        csv_write.writerow(["Avg",]+np.nanmean(result, 0).tolist())
