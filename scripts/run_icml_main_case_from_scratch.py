#!/usr/bin/env python3

import argparse
import importlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from accelerate.utils import set_seed
from PIL import Image

CURRENT_DIR = Path(__file__).resolve().parent
ROOT = CURRENT_DIR.parent
sys.path.append(str(ROOT))

from TRDI import AdaptiveTRDI, TRDI
from inversions.utils import is_float16
from matrics_calculator import MetricsCalculator


METHOD_MODULES = {
    "ddim": "inversions.unet_based.ddim",
    "renoise": "inversions.unet_based.renoise",
    "npi": "inversions.unet_based.npi",
    "gnri": "inversions.unet_based.gnri",
}

SCHEDULE_MODES = [
    "trdi",
    "adaptive_balanced",
    "adaptive_noise",
    "adaptive_noise_floor25",
    "adaptive_noise_floor50",
    "adaptive_noise_floor75",
    "adaptive_logsnr",
    "adaptive_curvature",
    "adaptive_late",
    "adaptive_match_balanced",
    "adaptive_match_noise",
    "adaptive_match_noise_floor25",
    "adaptive_match_noise_floor50",
    "adaptive_match_noise_floor75",
    "adaptive_match_logsnr",
    "adaptive_match_curvature",
    "adaptive_match_late",
]


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path, value):
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def clean_prompt(prompt):
    return prompt.replace("[", "").replace("]", "")


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
            "prompt_score": clean_prompt(mapping[key]["original_prompt"]),
            "prompt_editing_score": clean_prompt(mapping[key]["editing_prompt"]),
            "mask": mapping[key].get("mask"),
        }
        for key in selected
    ]


def parse_float_list(value):
    if value is None or value == "":
        return None
    return [float(item) for item in value.split(",")]


def parse_schedule_list(value):
    if value is None or value == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_candidate_image_roots(value):
    if value is None or value == "":
        return {}
    roots = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            schedule_mode, path = item.split("=", 1)
        elif ":" in item:
            schedule_mode, path = item.split(":", 1)
        else:
            raise ValueError(f"invalid candidate image root entry: {item}")
        roots[schedule_mode.strip()] = Path(path.strip())
    return roots


def mask_decode(encoded_mask, image_shape):
    length = image_shape[0] * image_shape[1]
    mask_array = np.zeros((length,), dtype=np.float32)
    for i in range(0, len(encoded_mask), 2):
        splice_len = min(encoded_mask[i + 1], length - encoded_mask[i])
        mask_array[encoded_mask[i] : encoded_mask[i] + splice_len] = 1.0
    mask_array = mask_array.reshape(image_shape[0], image_shape[1])
    mask_array[0, :] = 1.0
    mask_array[-1, :] = 1.0
    mask_array[:, 0] = 1.0
    mask_array[:, -1] = 1.0
    return mask_array[:, :, np.newaxis].repeat(3, axis=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_key", required=True)
    parser.add_argument("--method", choices=sorted(METHOD_MODULES), required=True)
    parser.add_argument("--model_type", choices=["SD15", "SD21", "SDXL", "SDXL_Turbo"], required=True)
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--num_inference_steps", type=int, required=True)
    parser.add_argument("--spacing", type=float, required=True)
    parser.add_argument("--trdi_window", type=int, required=True)
    parser.add_argument(
        "--schedule_mode",
        choices=SCHEDULE_MODES,
        default="trdi",
    )
    parser.add_argument("--inverse_schedule_mode", choices=SCHEDULE_MODES, default=None)
    parser.add_argument("--generation_schedule_mode", choices=SCHEDULE_MODES, default=None)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--inverse_guidance_scale", type=float, default=None)
    parser.add_argument("--negative_prompt_mode", choices=["original", "none"], default="original")
    parser.add_argument("--edit_guidance_rescale", type=float, default=0.0)
    parser.add_argument("--candidate_schedule_modes", default=None)
    parser.add_argument("--candidate_selection_mode", choices=["masked", "whole"], default="masked")
    parser.add_argument("--baseline_image_root", default=None)
    parser.add_argument("--candidate_image_roots", default=None)
    parser.add_argument("--selection_policy", choices=["score", "dominance"], default="score")
    parser.add_argument("--selection_clip_weight", type=float, default=1.0)
    parser.add_argument("--selection_preserve_weight", type=float, default=0.4)
    parser.add_argument("--selection_structure_weight", type=float, default=0.2)
    parser.add_argument("--selection_edit_clip_fraction", type=float, default=0.5)
    parser.add_argument("--selection_edit_clip_tolerance", type=float, default=None)
    parser.add_argument("--selection_baseline_switch_margin", type=float, default=None)
    parser.add_argument("--selection_min_target_clip_gain", type=float, default=None)
    parser.add_argument("--selection_target_clip_drop_tolerance", type=float, default=0.0)
    parser.add_argument("--selection_edit_clip_drop_tolerance", type=float, default=0.0)
    parser.add_argument("--selection_min_preserve_wins", type=int, default=5)
    parser.add_argument("--selection_min_preserve_wins_for_edit", type=int, default=0)
    parser.add_argument("--selection_max_structure_increase", type=float, default=None)
    parser.add_argument("--selection_max_mse_increase", type=float, default=None)
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


def build_trdi_timesteps(args):
    trdi = TRDI(args.num_inference_steps, spacing=args.spacing, window=args.trdi_window)
    timesteps = trdi.init_timesteps("leading")
    timesteps = trdi.rescaling_timesteps(timesteps)
    timesteps = trdi.reschedule(timesteps)
    return [int(t) for t in timesteps]


def build_timesteps(args, schedule_mode):
    if schedule_mode == "trdi":
        return build_trdi_timesteps(args)
    bounds = None
    density_mode = schedule_mode.replace("adaptive_", "")
    if density_mode.startswith("match_"):
        base_timesteps = build_trdi_timesteps(args)
        bounds = (min(base_timesteps), max(base_timesteps))
        density_mode = density_mode.replace("match_", "", 1)
    return [
        int(t)
        for t in AdaptiveTRDI(args.num_inference_steps, density_mode=density_mode).get_timesteps_adaptive(bounds=bounds)
    ]


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


def normalize_scores(values, higher_is_better=True):
    values = np.asarray(values, dtype=np.float64)
    span = float(values.max() - values.min()) if len(values) else 0.0
    if span <= 1e-12:
        scores = np.full_like(values, 0.5, dtype=np.float64)
    else:
        scores = (values - values.min()) / span
    if not higher_is_better:
        scores = 1.0 - scores
    return scores


def candidate_score_features(metrics_calculator, src_image, edited_image, prompt_editing, mask, selection_mode):
    use_mask = selection_mode == "masked" and mask is not None and np.asarray(mask).sum() > 0
    edit_mask = mask if use_mask else None
    preserve_mask = None
    if use_mask:
        preserve_mask = 1.0 - mask
        if preserve_mask.sum() <= 0:
            preserve_mask = None

    features = {
        "target_clip": metrics_calculator.calculate_clip_similarity(edited_image, prompt_editing, None),
        "target_clip_edit": None,
        "preserve_psnr": metrics_calculator.calculate_psnr(src_image, edited_image, preserve_mask, preserve_mask),
        "preserve_ssim": metrics_calculator.calculate_ssim(src_image, edited_image, preserve_mask, preserve_mask),
        "preserve_lpips": metrics_calculator.calculate_lpips(src_image, edited_image, preserve_mask, preserve_mask),
        "preserve_mse": metrics_calculator.calculate_mse(src_image, edited_image, preserve_mask, preserve_mask),
        "structure_distance": float(metrics_calculator.calculate_structure_distance(src_image, edited_image, None, None)),
    }
    if edit_mask is not None:
        features["target_clip_edit"] = metrics_calculator.calculate_clip_similarity(edited_image, prompt_editing, edit_mask)
    return features


def select_candidate(candidate_records, args):
    target_clip = normalize_scores([record["features"]["target_clip"] for record in candidate_records], True)
    if candidate_records[0]["features"]["target_clip_edit"] is None:
        clip_score = target_clip
    else:
        target_clip_edit = normalize_scores(
            [record["features"]["target_clip_edit"] for record in candidate_records], True
        )
        edit_fraction = min(max(args.selection_edit_clip_fraction, 0.0), 1.0)
        clip_score = (1.0 - edit_fraction) * target_clip + edit_fraction * target_clip_edit

    preserve_psnr = normalize_scores([record["features"]["preserve_psnr"] for record in candidate_records], True)
    preserve_ssim = normalize_scores([record["features"]["preserve_ssim"] for record in candidate_records], True)
    preserve_lpips = normalize_scores([record["features"]["preserve_lpips"] for record in candidate_records], False)
    preserve_mse = normalize_scores([record["features"]["preserve_mse"] for record in candidate_records], False)
    preserve_score = (preserve_psnr + preserve_ssim + preserve_lpips + preserve_mse) / 4.0
    structure_score = normalize_scores(
        [record["features"]["structure_distance"] for record in candidate_records], False
    )
    scores = (
        args.selection_clip_weight * clip_score
        + args.selection_preserve_weight * preserve_score
        + args.selection_structure_weight * structure_score
    )
    eligible = np.ones(len(candidate_records), dtype=bool)
    baseline_index = next(
        (index for index, record in enumerate(candidate_records) if record["schedule_mode"] == "trdi"),
        None,
    )
    if (
        args.selection_edit_clip_tolerance is not None
        and candidate_records[0]["features"]["target_clip_edit"] is not None
    ):
        edit_values = np.asarray(
            [record["features"]["target_clip_edit"] for record in candidate_records],
            dtype=np.float64,
        )
        best_edit = float(edit_values.max())
        eligible &= edit_values >= best_edit - args.selection_edit_clip_tolerance
    if baseline_index is not None:
        baseline_record = candidate_records[baseline_index]
        for index, record in enumerate(candidate_records):
            if index == baseline_index:
                eligible[index] = True
                continue
            if args.selection_min_target_clip_gain is not None:
                target_gain = record["features"]["target_clip"] - baseline_record["features"]["target_clip"]
                if target_gain < args.selection_min_target_clip_gain:
                    eligible[index] = False
            if args.selection_baseline_switch_margin is not None:
                feature_name = (
                    "target_clip_edit"
                    if baseline_record["features"]["target_clip_edit"] is not None
                    else "target_clip"
                )
                best_gain = record["features"][feature_name] - baseline_record["features"][feature_name]
                if best_gain < args.selection_baseline_switch_margin:
                    eligible[index] = False
            if args.selection_policy == "dominance":
                features = record["features"]
                baseline_features = baseline_record["features"]
                target_gain = features["target_clip"] - baseline_features["target_clip"]
                edit_gain = 0.0
                if baseline_features["target_clip_edit"] is not None:
                    edit_gain = features["target_clip_edit"] - baseline_features["target_clip_edit"]
                preserve_wins = 0
                preserve_wins += int(features["preserve_psnr"] > baseline_features["preserve_psnr"])
                preserve_wins += int(features["preserve_ssim"] > baseline_features["preserve_ssim"])
                preserve_wins += int(features["preserve_lpips"] < baseline_features["preserve_lpips"])
                preserve_wins += int(features["preserve_mse"] < baseline_features["preserve_mse"])
                preserve_wins += int(features["structure_distance"] < baseline_features["structure_distance"])
                structure_safe = (
                    args.selection_max_structure_increase is None
                    or features["structure_distance"]
                    <= baseline_features["structure_distance"] + args.selection_max_structure_increase
                )
                mse_safe = (
                    args.selection_max_mse_increase is None
                    or features["preserve_mse"]
                    <= baseline_features["preserve_mse"] + args.selection_max_mse_increase
                )
                semantic_ok = (
                    target_gain >= -args.selection_target_clip_drop_tolerance
                    and edit_gain >= -args.selection_edit_clip_drop_tolerance
                )
                improves_edit = (
                    edit_gain > 0.0
                    and preserve_wins >= args.selection_min_preserve_wins_for_edit
                )
                preserves = preserve_wins >= args.selection_min_preserve_wins
                if not structure_safe or not mse_safe or not semantic_ok or not (improves_edit or preserves):
                    eligible[index] = False
        if not eligible.any():
            eligible[baseline_index] = True
    best_index = int(np.argmax(np.where(eligible, scores, -np.inf)))
    if args.selection_baseline_switch_margin is not None and baseline_index is not None:
        if best_index != baseline_index:
            feature_name = (
                "target_clip_edit"
                if candidate_records[baseline_index]["features"]["target_clip_edit"] is not None
                else "target_clip"
            )
            best_gain = (
                candidate_records[best_index]["features"][feature_name]
                - candidate_records[baseline_index]["features"][feature_name]
            )
            if best_gain < args.selection_baseline_switch_margin:
                best_index = baseline_index
    for record, score in zip(candidate_records, scores):
        record["selection_score"] = float(score)
    return best_index


def main():
    args = parse_args()
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    set_seed(args.seed)

    output_dir = Path(args.output_root) / args.case_key
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "args.json"

    candidate_schedule_modes = parse_schedule_list(args.candidate_schedule_modes)
    candidate_image_roots = parse_candidate_image_roots(args.candidate_image_roots)
    if candidate_schedule_modes:
        invalid_modes = sorted(set(candidate_schedule_modes) - set(SCHEDULE_MODES))
        if invalid_modes:
            raise ValueError(f"invalid candidate schedule modes: {invalid_modes}")
        invalid_roots = sorted(set(candidate_image_roots) - set(candidate_schedule_modes))
        if invalid_roots:
            raise ValueError(f"candidate image roots must match candidate schedules: {invalid_roots}")
        candidate_timesteps = {
            schedule_mode: build_timesteps(args, schedule_mode) for schedule_mode in candidate_schedule_modes
        }
        inverse_schedule_mode = None
        generation_schedule_mode = None
        inverse_timesteps = None
        generation_timesteps = None
    else:
        candidate_timesteps = None
        inverse_schedule_mode = args.inverse_schedule_mode or args.schedule_mode
        generation_schedule_mode = args.generation_schedule_mode or args.schedule_mode
        inverse_timesteps = build_timesteps(args, inverse_schedule_mode)
        generation_timesteps = build_timesteps(args, generation_schedule_mode)
    data = load_dataset(args.annotation_mapping_file, args.start_index, args.max_samples)
    image_root = Path(args.annotation_image_root)
    baseline_image_root = Path(args.baseline_image_root) if args.baseline_image_root else None
    needs_pipeline = True
    if candidate_schedule_modes:
        needs_pipeline = False
        for datum in data:
            rel_path = datum["file"]
            for schedule_mode in candidate_schedule_modes:
                root_candidate_path = (
                    candidate_image_roots[schedule_mode] / rel_path
                    if schedule_mode in candidate_image_roots
                    else None
                )
                baseline_candidate_path = (
                    baseline_image_root / rel_path
                    if baseline_image_root is not None and schedule_mode == "trdi"
                    else None
                )
                if not (
                    root_candidate_path is not None
                    and root_candidate_path.is_file()
                    or baseline_candidate_path is not None
                    and baseline_candidate_path.is_file()
                ):
                    needs_pipeline = True
                    break
            if needs_pipeline:
                break
    metadata = {key: str(value) for key, value in vars(args).items()}
    metadata["inverse_schedule_mode"] = inverse_schedule_mode
    metadata["generation_schedule_mode"] = generation_schedule_mode
    metadata["inverse_timesteps"] = inverse_timesteps
    metadata["generation_timesteps"] = generation_timesteps
    metadata["candidate_schedule_modes"] = candidate_schedule_modes
    metadata["candidate_timesteps"] = candidate_timesteps
    metadata["candidate_image_roots"] = {key: str(value) for key, value in candidate_image_roots.items()}
    metadata["needs_pipeline"] = needs_pipeline
    metadata["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    pipe = load_pipeline(args) if needs_pipeline else None
    metrics_calculator = MetricsCalculator(args.device) if candidate_schedule_modes else None
    completed = 0
    skipped = 0
    failures = []
    selection_records_path = output_dir / "selection_records.json"
    selection_records_by_file = {}
    if candidate_timesteps and selection_records_path.is_file() and not args.overwrite:
        selection_records_by_file = {
            record["file"]: record for record in read_json(selection_records_path)
        }
    start = time.perf_counter()

    for index, datum in enumerate(data):
        rel_path = datum["file"]
        dst_path = output_dir / rel_path
        has_selection_record = rel_path in selection_records_by_file
        if (
            dst_path.is_file()
            and not args.overwrite
            and (not candidate_timesteps or has_selection_record)
        ):
            skipped += 1
            continue
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        image_file = image_root / rel_path
        if not image_file.is_file():
            raise FileNotFoundError(str(image_file))
        try:
            original_size = Image.open(image_file).size
            src_image = Image.open(image_file).convert("RGB")
            if candidate_timesteps:
                mask = None
                if datum.get("mask") is not None:
                    mask = mask_decode(datum["mask"], [src_image.size[1], src_image.size[0]])
                candidate_records = []
                for schedule_mode, timesteps in candidate_timesteps.items():
                    root_candidate_path = (
                        candidate_image_roots[schedule_mode] / rel_path
                        if schedule_mode in candidate_image_roots
                        else None
                    )
                    baseline_candidate_path = (
                        baseline_image_root / rel_path
                        if baseline_image_root is not None and schedule_mode == "trdi"
                        else None
                    )
                    reusable_source_path = None
                    if root_candidate_path is not None and root_candidate_path.is_file():
                        with Image.open(root_candidate_path) as source_candidate:
                            source_size = source_candidate.size
                            candidate_image = source_candidate.convert("RGB")
                        if source_size == original_size:
                            reusable_source_path = root_candidate_path
                        else:
                            candidate_image = candidate_image.resize(original_size)
                    elif baseline_candidate_path is not None and baseline_candidate_path.is_file():
                        with Image.open(baseline_candidate_path) as source_candidate:
                            source_size = source_candidate.size
                            candidate_image = source_candidate.convert("RGB")
                        if source_size == original_size:
                            reusable_source_path = baseline_candidate_path
                        else:
                            candidate_image = candidate_image.resize(original_size)
                    else:
                        inv_result = invert(pipe, args, str(image_file), datum["prompt"], timesteps)
                        candidate_image = edit(pipe, args, inv_result, datum["prompt"], datum["prompt_editing"], timesteps)
                        candidate_image = candidate_image.resize(original_size)
                    features = candidate_score_features(
                        metrics_calculator,
                        src_image,
                        candidate_image,
                        datum["prompt_editing_score"],
                        mask,
                        args.candidate_selection_mode,
                    )
                    candidate_records.append(
                        {
                            "schedule_mode": schedule_mode,
                            "image": candidate_image,
                            "source_path": reusable_source_path,
                            "features": features,
                        }
                    )
                    if args.device.startswith("cuda"):
                        torch.cuda.empty_cache()
                best_index = select_candidate(candidate_records, args)
                best_record = candidate_records[best_index]
                edited = best_record["image"]
                selected_source_path = best_record["source_path"]
                selection_record = {
                    "file": rel_path,
                    "selected_schedule_mode": best_record["schedule_mode"],
                    "candidates": [
                        {
                            "schedule_mode": record["schedule_mode"],
                            "features": record["features"],
                            "selection_score": record["selection_score"],
                        }
                        for record in candidate_records
                    ],
                }
                selection_records_by_file[rel_path] = selection_record
                if len(selection_records_by_file) % 10 == 0:
                    write_json_atomic(
                        selection_records_path,
                        [
                            selection_records_by_file[item["file"]]
                            for item in data
                            if item["file"] in selection_records_by_file
                        ],
                    )
                print(
                    f"[{args.case_key}] selected {best_record['schedule_mode']} for {rel_path}",
                    flush=True,
                )
            else:
                inv_result = invert(pipe, args, str(image_file), datum["prompt"], inverse_timesteps)
                edited = edit(pipe, args, inv_result, datum["prompt"], datum["prompt_editing"], generation_timesteps)
                edited = edited.resize(original_size)
                selected_source_path = None
            if selected_source_path is not None:
                shutil.copy2(selected_source_path, dst_path)
            else:
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
    if selection_records_by_file:
        write_json_atomic(
            selection_records_path,
            [
                selection_records_by_file[item["file"]]
                for item in data
                if item["file"] in selection_records_by_file
            ],
        )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
