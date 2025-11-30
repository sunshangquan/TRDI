import sys, os
# sys.path.append("../../")
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]  # 当前目录
config_path = CURRENT_DIR.rsplit('/', 2)[0]  # 上三级目录
sys.path.append(config_path)
from TRDI import TRDI
import os.path as osp
import argparse
import random, json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import lpips
import torch
from accelerate.utils import set_seed
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from inversions.unet_based.renoise import (SDInversionPipeline, SDXLInversionPipeline, CustomDDIMInversionScheduler,
                                        image2latents, latents2image)
from inversions.utils import pil2tensor, is_float16
from PIL import Image
from datasets import load_dataset

    
def read_json(file):
	with open(file) as f:
		df = json.load(f)
	return df



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["SD15", "SD21", "SDXL", 'SDXL_Turbo'],  default="SD15")
    parser.add_argument("--model_id", type=str, default=None)

    parser.add_argument("--output", type=str, default=None)

    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=0.0)

    parser.add_argument("--torch_dtype", type=torch.dtype, default=torch.float16)
    parser.add_argument("--variant", type=str, default="fp16")
    parser.add_argument("--data_name", type=str, default="coco")
    parser.add_argument("--path_name", type=str, default="./res_paper/")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--renoise_steps", type=int, default=1)
    parser.add_argument("--early_timestep", type=int, default=250)
    parser.add_argument("--early_weights", nargs="+", type=float, default=[0.5, 0.5])
    parser.add_argument("--latter_weights", nargs="+", type=float, default=[0.0, 1.0])
    parser.add_argument("--lambda_pair", type=int, default=10)
    parser.add_argument("--lambda_patch_kl", type=float, default=0.05)
    parser.add_argument("--num_reg_steps", type=int, default=4)
    parser.add_argument("--num_ac_rolls", type=int, default=5)


    parser.add_argument('--spacing', default=1.00, type=float)
    parser.add_argument('--trdi_window', type=int, default=1) 
    parser.add_argument("--save_src_path", type=str, default="input")

    args = parser.parse_args()

    return args

def inv(args, pipe, image_file, num_inference_steps, guidance_scale, prompt, timesteps, prompt_editing=None):
    inv_result = pipe.inverse(
        image=image_file,
        renoise_steps=args.renoise_steps,
        early_timestep=args.early_timestep,
        early_weights=args.early_weights,
        latter_weights=args.latter_weights,
        lambda_pair=args.lambda_pair,
        lambda_patch_kl=args.lambda_patch_kl,
        num_reg_steps=args.num_reg_steps,
        num_ac_rolls=args.num_ac_rolls,
        perform_enhance_edit=True,
        prompt=prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        timesteps=timesteps
    )
    ori_image = inv_result.ori_image
    
    recon_image = pipe(
        prompt=prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        latents=inv_result.zT,
        timesteps=timesteps
    ).images[0]
    return ori_image, recon_image

def main():
    args = parse_args()

    renoise_steps=args.renoise_steps,
    early_timestep=args.early_timestep,
    early_weights=args.early_weights,
    latter_weights=args.latter_weights,
    perform_enhance_edit=False,
    guidance_scale=args.guidance_scale,

    if args.seed is None:
        args.seed = random.randint(1, 10000)
    set_seed(args.seed)

    if args.output is None:
        args.output = str(Path(__file__).parent.parent.parent)

    assert args.model_type in ["SD15", "SD21", "SDXL", 'SDXL_Turbo']
    if args.model_id is None:
        if args.model_type == "SD15":
            args.model_id = "runwayml/stable-diffusion-v1-5"
        elif args.model_type == "SD21":
            args.model_id = "stabilityai/stable-diffusion-2-1"
        elif args.model_type == "SDXL":
            args.model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        elif args.model_type == 'SDXL_Turbo':
            args.model_id = "stabilityai/sdxl-turbo"

    args_text = "Args:\n"
    for k, v in vars(args).items():
        args_text += f"{k}: {v}\n"
    print(args_text)

    # if args.spacing == 1 and args.trdi_window == 0:
    #     timesteps = None
    # else:
    #     trdi = TRDI(args.num_inference_steps, spacing=args.spacing, window=args.trdi_window)
    #     timesteps = trdi.get_timesteps()
    #     timesteps = trdi.reschedule(timesteps)

    trdi = TRDI(args.num_inference_steps, spacing=args.spacing, window=args.trdi_window)
    timesteps = trdi.init_timesteps("leading")
    timesteps = trdi.rescaling_timesteps(timesteps)
    timesteps = trdi.reschedule(timesteps)

    # init pipeline
    scheduler = CustomDDIMInversionScheduler.from_pretrained(args.model_id, subfolder="scheduler", local_files_only=True)
    if args.model_type in ["SD15", "SD21"]:
        pipe = SDInversionPipeline.from_pretrained(
            args.model_id, 
            torch_dtype=args.torch_dtype if is_float16(args.model_type) else None, 
            variant=args.variant if is_float16(args.model_type) else None, 
            scheduler=scheduler,
            safety_checker = None,
            use_safetensors=True,
            local_files_only=True
        )
    else:
        pipe = SDXLInversionPipeline.from_pretrained(
            args.model_id, 
            torch_dtype=args.torch_dtype if is_float16(args.model_type) else None, 
            variant=args.variant if is_float16(args.model_type) else None, 
            scheduler=scheduler,
            safety_checker = None,
            use_safetensors=True,
            local_files_only=True
        )
    pipe.to(args.device)
    lpips_loss = lpips.LPIPS(net='alex')

    path_tar = os.path.join(args.path_name, args.data_name)
    os.makedirs(path_tar, exist_ok=True)
    path_tar = os.path.join(path_tar, "{}_ReNoise_{}_{}_{}_{}".format(args.model_type, args.spacing, args.num_inference_steps, args.trdi_window, args.guidance_scale), )
    os.makedirs(path_tar, exist_ok=True)
    ds = load_dataset("/data/.cache/huggingface/hub/datasets--lmms-lab--COCO-Caption2017/")#"lmms-lab/COCO-Caption2017")
    print(ds)
    
    
    with open(path_tar+"_timestep.txt", 'a+') as f:
        f.write("Steps: {}\n".format(pipe.scheduler.timesteps))
    # inference
    for i in range(len(ds['validation']['image'])):

        file = ds['validation']['question_id'][i]
        # img = ds['validation']['image'][i].convert("RGB")
        prompt = ds['validation']['answer'][i][0]
        prompt_editing = None

        print(i, file)

        print("pipe.scheduler.timesteps", pipe.scheduler.timesteps, len(pipe.scheduler.timesteps), path_tar)
        if i > 100:
            break
        num_inference_steps = args.num_inference_steps 
        guidance_scale = args.guidance_scale
        file = file
        print(i, path_tar+"/"+file, )
        if os.path.isfile(path_tar+"/"+file):
            continue
        if len(file.split("/")) > 1:
            os.makedirs(os.path.join(path_tar, "/".join(file.split("/")[:-1])), exist_ok=True)
        image_file = os.path.join("/data/COCO_val2017", file)
        original_size = Image.open(image_file).size

        ori_image, recon_image = inv(args, pipe, image_file, num_inference_steps, guidance_scale, prompt, timesteps, prompt_editing)

        psnr_score = psnr(np.array(ori_image), np.array(recon_image))
        ssim_score = ssim(np.array(ori_image), np.array(recon_image), win_size=11, channel_axis=2)
        lpips_score = lpips_loss(pil2tensor(ori_image), pil2tensor(recon_image)).item()
        print(f"[ReNoise] PSNR: {psnr_score:.2f}, SSIM: {ssim_score:.4f}, LPIPS: {lpips_score:.4f}")

        with open(path_tar+".txt", 'a+') as f:
            f.write(file+", PSNR: {}, SSIM: {}, LPIPS: {}\n".format(psnr_score, ssim_score, lpips_score))
        # rec_image = rec_image.resize((width, height))
        recon_image = recon_image.resize(original_size)
        recon_image.save(path_tar+"/"+file)
        if not os.path.isfile(os.path.join(args.save_src_path, file)):
            ori_image = ori_image.resize(original_size)
            ori_image.save(args.save_src_path+"/"+file)

        print("pipe.scheduler.timesteps", pipe.scheduler.timesteps, len(pipe.scheduler.timesteps))
        with open(path_tar+"_timestep.txt", 'a+') as f:
            f.write("Steps: {}\n".format(pipe.scheduler.timesteps))

if __name__ == "__main__":
    main()
