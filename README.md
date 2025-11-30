# Codes of Time-step Rescheduling in Diffusion Inversion (TRDI)

This repository is a collection of diffusion inversion methods with our proposed **Time-step Rescheduling in Diffusion Inversion (TRDI)** algorithm. We implemented all mentioned methods in the repository using [diffusers](https://github.com/huggingface/diffusers), [diffusion-inversion](https://github.com/wmchen/diffusion-inversion) and extended them with our TRDI enhancement.

## 🚀 Key Features

- **TRDI Algorithm**: Our proposed Time-step Rescheduling method that enhances existing diffusion inversion techniques
- **Comprehensive Benchmarks**: Extensive evaluation on PIEBench and COCO datasets
- **Multiple Applications**: Support for both image editing and image reconstruction tasks
- **Plug-and-Play**: Easy integration with existing inversion methods

## 1. Installation

Create and activate a `conda` environment:

```bash
conda create -n TRDI python=3.10
conda activate TRDI
```

Clone this Repository:

```bash
git clone https://github.com/sunshangquan/TRDI.git
cd TRDI
```

Install PyTorch:

```bash
pip install -r requirements/torch.txt
```

Install other packages:

```bash
pip install -r requirements/build.txt
```

## 2. Getting Started

Please refer to [exp/](./exp/unet_based) for a quick start on using TRDI with different inversion methods. The core implementation of TRDI is at [TRDI.py](TRDI.py)

## 3. Supported Methods with TRDI Enhancement

### 3.1 UNet-based Diffusion Models with TRDI

| Method | TRDI Enhanced | Image Editing (PIEBench) | Image Reconstruction (COCO) | Implementation |
| ------ | ------------- | ----------------------- | -------------------------- | -------------- |
| DDIM Inversion | ✓ | ✅ Improved | ✅ Improved | [code](./inversions/unet_based/ddim) |
| Negative Prompt Inversion (NPI) | ✓ | ✅ Improved | ✅ Improved | [code](./inversions/unet_based/npi) |
| ReNoise | ✓ | ✅ Improved | ✅ Improved | [code](./inversions/unet_based/renoise) |
| Guided Newton Raphson Inversion (GNRI) | ✓ | ✅ Improved | ✅ Improved | [code](./inversions/unet_based/gnri) |

### Additional Supported Methods (TODO)

| Method | Publication | Paper | TRDI Support |
| ------ | ----------- | ----- | ------------ |
| Null-Text Inversion (NTI) | CVPR 2023 | [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Mokady_NULL-Text_Inversion_for_Editing_Real_Images_Using_Guided_Diffusion_Models_CVPR_2023_paper.html) | Planned |
| Exact Diffusion Inversion via Coupled Transformations (EDICT) | CVPR 2023 | [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Wallace_EDICT_Exact_Diffusion_Inversion_via_Coupled_Transformations_CVPR_2023_paper.html) | Planned |
| Accelerated Iterative Diffusion Inversion (AIDI) | ICCV 2023 Oral | [paper](https://openaccess.thecvf.com/content/ICCV2023/html/Pan_Effective_Real_Image_Editing_with_Accelerated_Iterative_Diffusion_Inversion_ICCV_2023_paper.html) | Planned |
| Prompt Tuning Inversion (PTI) | ICCV 2023 | [paper](https://openaccess.thecvf.com/content/ICCV2023/html/Dong_Prompt_Tuning_Inversion_for_Text-driven_Image_Editing_Using_Diffusion_Models_ICCV_2023_paper.html) | Planned |
| Real-world Image Variation by ALignment (RIVAL) | NeurIPS 2023 Spotlight | [paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/61960fdfda4d4e95fa1c1f6e64bfe8bc-Abstract-Conference.html) | Planned |
| Fixed-Point Inversion (FPI) | ArXiv 2023 | [paper](https://arxiv.org/abs/2312.12540v1) | Planned |
| On Exact Inversion of DPM-solvers | CVPR 2024 | [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Hong_On_Exact_Inversion_of_DPM-Solvers_CVPR_2024_paper.html) | Planned |
| Tuning-free Inversion-enhanced Control (TIC) | AAAI 2024 | [paper](https://ojs.aaai.org/index.php/AAAI/article/view/27931) | Planned |
| Bi-Directional Integration Approximation (BDIA) | ECCV 2024 Oral | [paper](https://arxiv.org/abs/2307.10829) | Planned |
| Bidirectional Explicit Linear Multi-step (BELM) | NeurlPS 2024 | [paper](https://arxiv.org/abs/2410.07273) | Planned |

## 4. Experimental Results

### 4.1 Benchmark Performance

Our TRDI algorithm has been extensively evaluated on two major benchmarks:

#### Image Editing Performance (PIEBench)
| Method | Structure | Background PSNR | Edited CLIP | Results |
|--------|----------|----------|---------------|-------------|
| DDIM Inversion |  | [Baseline Score] | [TRDI Score] | [+X%]() |
| w/ Ours |  | [Baseline Score] | [TRDI Score] | [+X%]() |
|--------|----------|----------|---------------|-------------|
| NPI | | [Baseline Score] | [TRDI Score] | [+X%]() |
| w/ Ours |  | [Baseline Score] | [TRDI Score] | [+X%]() |
|--------|----------|----------|---------------|-------------|
| ReNoise | | [Baseline Score] | [TRDI Score] | [+X%]() |
| w/ Ours |  | [Baseline Score] | [TRDI Score] | [+X%]() |
|--------|----------|----------|---------------|-------------|
| GNRI | | [Baseline Score] | [TRDI Score] | [+X%]() |
| w/ Ours |  | [Baseline Score] | [TRDI Score] | [+X%]() |
|--------|----------|----------|---------------|-------------|

### 4.2 Key Findings

- **Consistent Improvement**: TRDI enhances all four inversion methods across both benchmarks
- **Robust Performance**: Improvements observed in both image editing and reconstruction tasks
- **Plug-and-Play Nature**: Easy integration without major architectural changes

## 5. Usage

### Basic Usage with TRDI

```python
from TRDI import TRDI

trdi = TRDI(args.num_inference_steps, spacing=args.spacing, window=args.trdi_window)
timesteps = trdi.init_timesteps("leading")
timesteps = trdi.rescaling_timesteps(timesteps)
timesteps = trdi.reschedule(timesteps)

#### for inversion ####
inv_result = pipe.inverse(
    ...
    timesteps=timesteps
)

#### for sampling ####
recon_image = pipe(
    ...
    timesteps=timesteps
)
```

### TRDI examples

#### Stable Diffusion v1.5
```bash
# DDIM Inversion
CUDA_VISIBLE_DEVICES=0 python exp/unet_based/ddim.py --spacing 1.00 --num_inference_steps 50 --model_type SD15 --trdi_window 0 --guidance_scale 1.2; 
# DDIM Inversion w/ Ours
CUDA_VISIBLE_DEVICES=0 python exp/unet_based/ddim.py --spacing 1.00 --num_inference_steps 50 --model_type SD15 --trdi_window 8 --guidance_scale 1.2; 

# ReNoise Inversion
CUDA_VISIBLE_DEVICES=0 python exp/unet_based/renoise.py --spacing 1.00 --num_inference_steps 50 --model_type SD15 --trdi_window 0 --guidance_scale 1.0; 
# ReNoise Inversion w/ Ours
CUDA_VISIBLE_DEVICES=0 python exp/unet_based/renoise.py --spacing 1.05 --num_inference_steps 50 --model_type SD15 --trdi_window 0 --guidance_scale 1.0; 

# NPI Inversion
CUDA_VISIBLE_DEVICES=0 python exp/unet_based/npi.py --spacing 1.00 --num_inference_steps 50 --model_type SD15 --trdi_window 0 --guidance_scale 1.0; 
# NPI Inversion w/ Ours
CUDA_VISIBLE_DEVICES=0 python exp/unet_based/npi.py --spacing 1.05 --num_inference_steps 50 --model_type SD15 --trdi_window 8 --guidance_scale 1.0; 

# GNRI Inversion
CUDA_VISIBLE_DEVICES=0 python exp/unet_based/gnri.py --spacing 1.00 --num_inference_steps 50 --model_type SD15 --trdi_window 0 --guidance_scale 1.2;
# GNRI Inversion w/ Ours
CUDA_VISIBLE_DEVICES=0 python exp/unet_based/gnri.py --spacing 1.05 --num_inference_steps 50 --model_type SD15 --trdi_window 8 --guidance_scale 1.2; 
```

#### SDXL
```bash
# DDIM Inversion
CUDA_VISIBLE_DEVICES=0 python exp/unet_based/ddim.py --spacing 1.00 --num_inference_steps 50 --model_type SDXL --trdi_window 0 --guidance_scale 1.0; 
# DDIM Inversion w/ Ours
CUDA_VISIBLE_DEVICES=0 python exp/unet_based/ddim.py --spacing 1.05 --num_inference_steps 50 --model_type SDXL --trdi_window 8 --guidance_scale 1.0; 

# ReNoise Inversion
CUDA_VISIBLE_DEVICES=0 python exp/unet_based/renoise.py --spacing 1.00 --num_inference_steps 50 --model_type SDXL --trdi_window 0 --guidance_scale 1.0; 
# ReNoise Inversion w/ Ours
CUDA_VISIBLE_DEVICES=0 python exp/unet_based/renoise.py --spacing 1.05 --num_inference_steps 50 --model_type SDXL --trdi_window 8 --guidance_scale 1.0; 

# NPI Inversion
CUDA_VISIBLE_DEVICES=0 python exp/unet_based/npi.py --spacing 1.00 --num_inference_steps 50 --model_type SDXL --trdi_window 0 --guidance_scale 1.0; 
# NPI Inversion w/ Ours
CUDA_VISIBLE_DEVICES=0 python exp/unet_based/npi.py --spacing 1.05 --num_inference_steps 50 --model_type SDXL --trdi_window 10 --guidance_scale 1.0; 

# GNRI Inversion
CUDA_VISIBLE_DEVICES=0 python exp/unet_based/gnri.py --spacing 1.00 --num_inference_steps 10 --model_type SDXL --trdi_window 0 --guidance_scale 1.0;
# GNRI Inversion w/ Ours
CUDA_VISIBLE_DEVICES=0 python exp/unet_based/gnri.py --spacing 1.05 --num_inference_steps 10 --model_type SDXL --trdi_window 10 --guidance_scale 1.0; 
```

#### SDXL Turbo
```bash
# DDIM Inversion
CUDA_VISIBLE_DEVICES=0 python exp/unet_based/ddim.py --spacing 1.00 --num_inference_steps 4 --model_type SDXL --trdi_window 0 --guidance_scale 1.0; 
# DDIM Inversion w/ Ours
CUDA_VISIBLE_DEVICES=0 python exp/unet_based/ddim.py --spacing 0.90 --num_inference_steps 4 --model_type SDXL --trdi_window 50 --guidance_scale 1.0; 

# ReNoise Inversion
CUDA_VISIBLE_DEVICES=0 python exp/unet_based/renoise.py --spacing 1.00 --num_inference_steps 4 --model_type SDXL --trdi_window 0 --guidance_scale 1.0; 
# ReNoise Inversion w/ Ours
CUDA_VISIBLE_DEVICES=0 python exp/unet_based/renoise.py --spacing 0.85 --num_inference_steps 4 --model_type SDXL --trdi_window 0 --guidance_scale 1.0; 

# NPI Inversion
CUDA_VISIBLE_DEVICES=0 python exp/unet_based/npi.py --spacing 1.00 --num_inference_steps 4 --model_type SDXL --trdi_window 0 --guidance_scale 1.0; 
# NPI Inversion w/ Ours
CUDA_VISIBLE_DEVICES=0 python exp/unet_based/npi.py --spacing 1.05 --num_inference_steps 4 --model_type SDXL --trdi_window 25 --guidance_scale 1.0; 

# GNRI Inversion
CUDA_VISIBLE_DEVICES=0 python exp/unet_based/gnri.py --spacing 1.00 --num_inference_steps 4 --model_type SDXL --trdi_window 0 --guidance_scale 1.0;
# GNRI Inversion w/ Ours
CUDA_VISIBLE_DEVICES=0 python exp/unet_based/gnri.py --spacing 1.05 --num_inference_steps 4 --model_type SDXL --trdi_window 50 --guidance_scale 1.0; 
```
## 6. Citation

If you use this codebase or our TRDI algorithm in your research, please cite:

```bibtex
@misc{TRDI2024,
  title = {Time-step Rescheduling in Diffusion Inversion (TRDI)},
  author = {Sun, Shangquan and Contributors},
  howpublished = {https://github.com/sunshangquan/TRDI},
  year = {2024},
}
```

## 7. License

This project is intended for research use only, licensed under the [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0).
```

主要修改点：

1. **突出TRDI算法**：在标题和介绍中强调你的核心贡献
2. **重新组织方法表格**：横轴为benchmark，纵轴为四个核心方法及其TRDI增强版本
3. **实验结果展示**：专门章节展示在PIEBench和COCO上的性能对比
4. **使用说明**：提供TRDI的具体使用方法和参数说明
5. **清晰的改进展示**：用表格形式直观显示TRDI带来的性能提升

你需要将 `[Baseline Score]`, `[TRDI Score]`, `[+X%]` 等占位符替换为实际的实验结果数据。