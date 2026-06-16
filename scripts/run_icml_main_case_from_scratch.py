#!/usr/bin/env python3

import argparse
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
from accelerate.utils import set_seed
from PIL import Image

CURRENT_DIR = Path(__file__).resolve().parent
ROOT = CURRENT_DIR.parent
sys.path.append(str(ROOT))

from TRDI import TRDI
from inversions.utils import is_float16


METHOD_MODULES = {
    "ddim": "inversions.unet_based.ddim",
    "renoise": "inversions.unet_based.renoise",
    "npi": "inversions.unet_based.npi",
    "gnri": "inversions.unet_based.gnri",
}


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_dataset(mapping_path, start_index, max_samples):
    mapping = read_json(mapping_path)
    keys = list(mapping.keys())
    if max_samples is None:
        selected = keys[start_index:]
    else:
        selected = keys[start_index : start_index + max_samples]
    return [
        {
            "file": mapping[key]["image_path"],
            "prompt": mapping[key]["original_prompt"],
            "prompt_editing": mapping[key]["editing_prompt"],
        }
        for key in selected
    ]


def parse_float_list(value):
    if value is None or value == "":
        return None
    return [float(item) for item in value.split(",")]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_key", required=True)
    parser.add_argument("--method", choices=sorted(METHOD_MODULES), required=True)
    parser.add_argument("--model_type", choices=["SD15", "SD21", "SDXL", "SDXL_Turbo"], required=True)
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--num_inference_steps", type=int, required=True)
    parser.add_argument("--spacing", type=float, required=True)
    parser.add_argument("--trdi_window", type=int, required=True)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--inverse_guidance_scale", type=float, default=None)
    parser.add_argument("--negative_prompt_mode", choices=["original", "none"], default="original")
    parser.add_argument("--edit_guidance_rescale", type=float, default=0.0)
    parser.add_argument("--torch_dtype", choices=["float16", "float32", "bfloat16"], default="float16")
    parser.add_argument("--variant", default="fp16")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--annotation_mapping_file", required=True)
    parser.add_argument("--annotation_image_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
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


def resolve_model_id(args):
    if args.model_id:
        return args.model_id
    if args.model_type == "SD15":
        return "stable-diffusion-v1-5/stable-diffusion-v1-5"
    if args.model_type == "SD21":
        return "stabilityai/stable-diffusion-2-1"
    if args.model_type == "SDXL":
        return "stabilityai/stable-diffusion-xl-base-1.0"
    return "stabilityai/sdxl-turbo"


def resolve_dtype(name):
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(name)


def resolve_variant(args, torch_dtype):
    if torch_dtype != torch.float16:
        return None
    return args.variant if is_float16(args.model_type) else None


def build_timesteps(args):
    trdi = TRDI(args.num_inference_steps, spacing=args.spacing, window=args.trdi_window)
    timesteps = trdi.init_timesteps("leading")
    timesteps = trdi.rescaling_timesteps(timesteps)
    timesteps = trdi.reschedule(timesteps)
    return [int(t) for t in timesteps]


def load_pipeline(args):
    module = importlib.import_module(METHOD_MODULES[args.method])
    model_id = resolve_model_id(args)
    torch_dtype = resolve_dtype(args.torch_dtype)
    variant = resolve_variant(args, torch_dtype)
    scheduler = module.CustomDDIMInversionScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        local_files_only=True,
    )
    pipe_cls = module.SDInversionPipeline if args.model_type in {"SD15", "SD21"} else module.SDXLInversionPipeline
    common_kwargs = {
        "torch_dtype": torch_dtype,
        "scheduler": scheduler,
        "safety_checker": None,
        "use_safetensors": True,
        "local_files_only": True,
    }
    try:
        pipe = pipe_cls.from_pretrained(model_id, variant=variant, **common_kwargs)
    except ValueError as exc:
        if variant is None or "variant=fp16" not in str(exc):
            raise
        pipe = pipe_cls.from_pretrained(model_id, variant=None, **common_kwargs)
    pipe.to(args.device)
    return pipe


def invert(pipe, args, image_file, prompt, timesteps):
    if args.method == "ddim":
        return pipe.inverse(
            image=image_file,
            prompt=prompt,
            guidance_scale=0.0,
            num_inference_steps=args.num_inference_steps,
            timesteps=timesteps,
        )
    if args.method == "renoise":
        return pipe.inverse(
            image=image_file,
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
    if args.method == "gnri":
        return pipe.inverse(
            image=image_file,
            prompt=prompt,
            guidance_scale=args.inverse_guidance_scale if args.inverse_guidance_scale is not None else args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            timesteps=timesteps,
            num_iter_steps=args.num_iter_steps,
            param_lambda=args.param_lambda,
            param_eta=args.param_eta,
        )
    return pipe.inverse(
        image=image_file,
        prompt=prompt,
        guidance_scale=args.inverse_guidance_scale if args.inverse_guidance_scale is not None else args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        timesteps=timesteps,
    )


def edit(pipe, args, inv_result, prompt, prompt_editing, timesteps):
    kwargs = {
        "prompt": prompt_editing,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "guidance_rescale": args.edit_guidance_rescale,
        "latents": inv_result.zT,
        "timesteps": timesteps,
    }
    if args.method != "renoise" and args.negative_prompt_mode == "original":
        kwargs["negative_prompt"] = prompt
    return pipe(**kwargs).images[0]


def main():
    args = parse_args()
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    set_seed(args.seed)

    output_dir = Path(args.output_root) / args.case_key
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "args.json"

    timesteps = build_timesteps(args)
    metadata = {key: str(value) for key, value in vars(args).items()}
    metadata["timesteps"] = timesteps
    metadata["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    pipe = load_pipeline(args)
    data = load_dataset(args.annotation_mapping_file, args.start_index, args.max_samples)
    image_root = Path(args.annotation_image_root)
    completed = 0
    skipped = 0
    failures = []
    start = time.perf_counter()

    for index, datum in enumerate(data):
        rel_path = datum["file"]
        dst_path = output_dir / rel_path
        if dst_path.is_file() and not args.overwrite:
            skipped += 1
            continue
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        image_file = image_root / rel_path
        if not image_file.is_file():
            raise FileNotFoundError(str(image_file))
        try:
            original_size = Image.open(image_file).size
            inv_result = invert(pipe, args, str(image_file), datum["prompt"], timesteps)
            edited = edit(pipe, args, inv_result, datum["prompt"], datum["prompt_editing"], timesteps)
            edited = edited.resize(original_size)
            edited.save(dst_path)
            completed += 1
            elapsed = time.perf_counter() - start
            print(f"[{args.case_key}] {index + 1}/{len(data)} saved {rel_path} elapsed={elapsed:.1f}s", flush=True)
        except Exception as exc:
            failures.append({"file": rel_path, "error": repr(exc)})
            print(f"[{args.case_key}] failed {rel_path}: {exc!r}", flush=True)
            raise

    summary = {
        "case_key": args.case_key,
        "completed": completed,
        "skipped": skipped,
        "failures": failures,
        "elapsed_seconds": time.perf_counter() - start,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
