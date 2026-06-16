#!/usr/bin/env python3

import argparse
import csv
import importlib
import json
import math
import random
import sys
import time
from pathlib import Path

import lpips
import numpy as np
import torch
from accelerate.utils import set_seed
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from TRDI import TRDI  # noqa: E402
from inversions.utils import is_float16, pil2tensor  # noqa: E402


MODEL_IDS = {
    "SD15": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "SD21": "stabilityai/stable-diffusion-2-1",
    "SDXL": "stabilityai/stable-diffusion-xl-base-1.0",
    "SDXL_Turbo": "stabilityai/sdxl-turbo",
}


def parse_float_list(value):
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [float(item) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_key", required=True)
    parser.add_argument("--method", choices=["ddim", "renoise", "npi", "gnri"], required=True)
    parser.add_argument("--model_type", choices=["SD15", "SD21", "SDXL", "SDXL_Turbo"], default="SD15")
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--spacing", type=float, default=1.0)
    parser.add_argument("--trdi_window", type=int, default=0)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--inverse_guidance_scale", type=float, default=None)
    parser.add_argument("--manifest_path", default=str(ROOT / "data/manifests/coco_input_subset_100.json"))
    parser.add_argument("--image_root", default=str(ROOT / "input"))
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--variant", default="fp16")
    parser.add_argument("--renoise_steps", type=int, default=1)
    parser.add_argument("--early_timestep", type=int, default=250)
    parser.add_argument("--early_weights", default="0.5,0.5")
    parser.add_argument("--latter_weights", default="0.0,1.0")
    parser.add_argument("--lambda_pair", type=int, default=10)
    parser.add_argument("--lambda_patch_kl", type=float, default=0.05)
    parser.add_argument("--num_reg_steps", type=int, default=4)
    parser.add_argument("--num_ac_rolls", type=int, default=5)
    parser.add_argument("--num_iter_steps", type=int, default=2)
    parser.add_argument("--param_lambda", type=float, default=0.1)
    parser.add_argument("--param_eta", type=float, default=0.0)
    return parser.parse_args()


def get_timesteps(num_inference_steps, spacing, window):
    trdi = TRDI(num_inference_steps, spacing=spacing, window=window)
    timesteps = trdi.init_timesteps("leading")
    timesteps = trdi.rescaling_timesteps(timesteps)
    return trdi.reschedule(timesteps)


def load_pipeline(args):
    module = importlib.import_module(f"inversions.unet_based.{args.method}")
    scheduler = module.CustomDDIMInversionScheduler.from_pretrained(
        args.model_id, subfolder="scheduler", local_files_only=True
    )
    dtype = torch.float16 if args.torch_dtype == "float16" else torch.float32
    pipe_cls = module.SDInversionPipeline if args.model_type in ["SD15", "SD21"] else module.SDXLInversionPipeline
    load_kwargs = dict(
        torch_dtype=dtype if is_float16(args.model_type) else None,
        variant=args.variant if is_float16(args.model_type) else None,
        scheduler=scheduler,
        safety_checker=None,
        use_safetensors=True,
        local_files_only=True,
    )
    try:
        pipe = pipe_cls.from_pretrained(args.model_id, **load_kwargs)
    except ValueError as exc:
        if "variant=fp16" not in str(exc):
            raise
        load_kwargs["variant"] = None
        pipe = pipe_cls.from_pretrained(args.model_id, **load_kwargs)
    return pipe.to(args.device)


def inverse_and_reconstruct(args, pipe, image_path, prompt, timesteps):
    if args.method == "renoise":
        inv_result = pipe.inverse(
            image=str(image_path),
            renoise_steps=args.renoise_steps,
            early_timestep=args.early_timestep,
            early_weights=parse_float_list(args.early_weights),
            latter_weights=parse_float_list(args.latter_weights),
            lambda_pair=args.lambda_pair,
            lambda_patch_kl=args.lambda_patch_kl,
            num_reg_steps=args.num_reg_steps,
            num_ac_rolls=args.num_ac_rolls,
            perform_enhance_edit=True,
            prompt=prompt,
            guidance_scale=args.inverse_guidance_scale if args.inverse_guidance_scale is not None else args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            timesteps=timesteps,
        )
    else:
        inverse_guidance = 0.0 if args.method == "ddim" else args.guidance_scale
        if args.inverse_guidance_scale is not None:
            inverse_guidance = args.inverse_guidance_scale
        inverse_kwargs = {}
        if args.method == "gnri":
            inverse_kwargs.update(
                num_iter_steps=args.num_iter_steps,
                param_lambda=args.param_lambda,
                param_eta=args.param_eta,
            )
        inv_result = pipe.inverse(
            image=str(image_path),
            prompt=prompt,
            guidance_scale=inverse_guidance,
            num_inference_steps=args.num_inference_steps,
            timesteps=timesteps,
            **inverse_kwargs,
        )
    recon = pipe(
        prompt=prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        latents=inv_result.zT,
        timesteps=timesteps,
    ).images[0]
    return inv_result.ori_image, recon


def resolve_image_path(record, image_root):
    path = Path(record["image_path"])
    if path.is_file():
        return path
    return image_root / path.name


def main():
    args = parse_args()
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    set_seed(args.seed)
    if args.model_id is None:
        args.model_id = MODEL_IDS[args.model_type]

    image_root = Path(args.image_root)
    output_dir = Path(args.output_root) / args.case_key / "images"
    eval_dir = Path(args.eval_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    (Path(args.output_root) / args.case_key).mkdir(parents=True, exist_ok=True)
    (Path(args.output_root) / args.case_key / "args.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )

    records = json.loads(Path(args.manifest_path).read_text(encoding="utf-8"))
    if args.max_samples is not None:
        records = records[: args.max_samples]

    timesteps = get_timesteps(args.num_inference_steps, args.spacing, args.trdi_window)
    pipe = load_pipeline(args)
    lpips_loss = lpips.LPIPS(net="alex")

    csv_path = eval_dir / f"{args.case_key}.csv"
    rows = []
    start = time.time()
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_id", "psnr", "ssim", "lpips"])
        for index, record in enumerate(records, start=1):
            image_path = resolve_image_path(record, image_root)
            prompt = record["prompt"]
            file_id = record["id"]
            out_path = output_dir / f"{file_id}.jpg"
            ori_image, recon_image = inverse_and_reconstruct(args, pipe, image_path, prompt, timesteps)
            original_size = Image.open(image_path).size
            recon_image = recon_image.resize(original_size)
            ori_image = ori_image.resize(original_size)
            recon_image.save(out_path)

            psnr_score = float(psnr(np.asarray(ori_image), np.asarray(recon_image)))
            ssim_score = float(ssim(np.asarray(ori_image), np.asarray(recon_image), win_size=11, channel_axis=2))
            lpips_score = float(lpips_loss(pil2tensor(ori_image), pil2tensor(recon_image)).item())
            row = [file_id, psnr_score, ssim_score, lpips_score]
            writer.writerow(row)
            handle.flush()
            rows.append(row[1:])
            print(
                f"[{args.case_key}] {index}/{len(records)} saved {out_path.name} "
                f"PSNR={psnr_score:.4f} SSIM={ssim_score:.4f} LPIPS={lpips_score:.4f} "
                f"elapsed={time.time() - start:.1f}s",
                flush=True,
            )
        avg = np.asarray(rows, dtype=np.float32)
        avg_values = np.nanmean(avg, axis=0).tolist() if len(rows) else [math.nan, math.nan, math.nan]
        writer.writerow(["Avg", *avg_values])
    print(f"[{args.case_key}] wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
