# Reproduction Guide

This document describes the maintained reproduction entry points. The scripts avoid hard-coded local paths; pass dataset and output paths explicitly.

## PIE-Bench Editing

`scripts/run_icml_main_case_from_scratch.py` runs one editing case and writes generated images plus `args.json`.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_icml_main_case_from_scratch.py \
  --case_key sdxl_ddim_trdi \
  --method ddim \
  --model_type SDXL \
  --num_inference_steps 50 \
  --spacing 1.035 \
  --trdi_window 9 \
  --guidance_scale 1.025 \
  --annotation_mapping_file /path/to/piebench_mapping.json \
  --annotation_image_root /path/to/annotation_images \
  --output_root outputs/piebench
```

The mapping JSON should map sample IDs to records with:

```json
{
  "sample_id": {
    "image_path": "relative/path/to/image.png",
    "original_prompt": "source prompt",
    "editing_prompt": "target prompt"
  }
}
```

Evaluate editing outputs with `evaluate_my.py`:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_my.py \
  --annotation_mapping_file /path/to/piebench_mapping.json \
  --src_image_folder /path/to/annotation_images \
  --metrics structure_distance psnr_unedit_part lpips_unedit_part mse_unedit_part ssim_unedit_part clip_similarity_target_image clip_similarity_target_image_edit_part \
  --tgt_path outputs/piebench \
  --tgt_methods sdxl_ddim_trdi \
  --result_path outputs/evaluation \
  --annotation_images
```

If the CLIP model is already downloaded locally, set:

```bash
export CLIP_MODEL_NAME_OR_PATH=/path/to/openai/clip-vit-large-patch14
```

## COCO-Style Reconstruction

Create a manifest with `scripts/prepare_coco_caption_subset.py`:

```bash
python scripts/prepare_coco_caption_subset.py \
  --captions_json /path/to/captions_val2017.json \
  --image_dir /path/to/val2017 \
  --output data/manifests/coco_subset.json \
  --max_samples 100
```

Run one reconstruction case:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_icml_recon_case_from_scratch.py \
  --case_key sd15_ddim_trdi \
  --method ddim \
  --model_type SD15 \
  --num_inference_steps 50 \
  --spacing 1.05 \
  --trdi_window 8 \
  --guidance_scale 2.0 \
  --manifest_path data/manifests/coco_subset.json \
  --image_root /path/to/images \
  --output_root outputs/recon \
  --eval_dir outputs/recon_eval
```

## Table Builders

The table builders read evaluation CSV files and optionally compare them against LaTeX tables from the paper package.

```bash
python scripts/build_icml_main_table_from_run.py \
  --paper_table /path/to/result.tex \
  --eval_dir outputs/evaluation \
  --output_md outputs/main_table.md \
  --output_csv outputs/main_table.csv
```

```bash
python scripts/build_icml_recon_table_from_run.py \
  --paper_table /path/to/recon.tex \
  --eval_dir outputs/recon_eval \
  --output_md outputs/recon_table.md \
  --output_csv outputs/recon_table.csv
```

## Notes

- Model weights are loaded with `local_files_only=True`. Download the required Hugging Face models before running offline experiments, or adapt the script if online loading is desired.
- Generated outputs, datasets, checkpoints, logs, and local search scripts are intentionally excluded from source control.
