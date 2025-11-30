# DDIM
for model in SD15; do 
for spacing in 1.0; do 
for window in 0 10; do 
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /home/sunshangquan/anaconda3/envs/p2p/bin/python evaluate_coco.py \
    --metrics "psnr" "ssim" "lpips" "mse" --tgt_path input --tgt_methods "/data/proj1/diffusion-inversion/res_paper/coco/"$model"_DDIM_"$spacing"_50_"$window"_2.0" ;
done
done
done


# NPI
for model in SD15; do 
for spacing in 1.0; do 
for window in 0 10; do 
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /home/sunshangquan/anaconda3/envs/p2p/bin/python evaluate_coco.py \
    --metrics "psnr" "ssim" "lpips" "mse" --tgt_path input --tgt_methods "/data/proj1/diffusion-inversion/res_paper/coco/"$model"_NPI_"$spacing"_50_"$window"_2.0" ;
done
done
done

# ReNoise
for model in SD15; do 
for spacing in 1.0; do 
for window in 0 10; do 
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /home/sunshangquan/anaconda3/envs/p2p/bin/python evaluate_coco.py \
    --metrics "psnr" "ssim" "lpips" "mse" --tgt_path input --tgt_methods "/data/proj1/diffusion-inversion/res_paper/coco/"$model"_ReNoise_"$spacing"_50_"$window"_2.0" ;
done
done
done


# GNRI
for model in SD15; do 
for spacing in 1.0; do 
for window in 0 10; do 
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /home/sunshangquan/anaconda3/envs/p2p/bin/python evaluate_coco.py \
    --metrics "psnr" "ssim" "lpips" "mse" --tgt_path input --tgt_methods "/data/proj1/diffusion-inversion/res_paper/coco/"$model"_GNRI_"$spacing"_50_"$window"_1.5" ;
done
done
done

############################################################

# DDIM
for model in SDXL_Turbo; do 
for spacing in 0.8 0.85 0.9 0.95 1.0 1.05 1.1 1.15 1.2; do 
for window in 0 10 25 35 50; do 
for guidance_scale in 0.0 1.0 1.5; do 
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /home/sunshangquan/anaconda3/envs/p2p/bin/python evaluate_coco.py \
    --metrics "psnr" "ssim" "lpips" "mse" --tgt_path input --tgt_methods "/data/proj1/diffusion-inversion/res_paper/coco/"$model"_DDIM_"$spacing"_4_"$window"_"$guidance_scale ;
done
done
done
done

# NPI
for model in SDXL_Turbo; do 
for spacing in 0.8 0.85 0.9 0.95 1.0 1.05 1.1 1.15 1.2; do 
for window in 0 10 25 35 50; do 
for guidance_scale in 0.0 1.0 1.5; do 
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /home/sunshangquan/anaconda3/envs/p2p/bin/python evaluate_coco.py \
    --metrics "psnr" "ssim" "lpips" "mse" --tgt_path input --tgt_methods "/data/proj1/diffusion-inversion/res_paper/coco/"$model"_NPI_"$spacing"_4_"$window"_"$guidance_scale ;
done
done
done
done

# ReNoise
for model in SDXL_Turbo; do 
for spacing in 0.8 0.85 0.9 0.95 1.0 1.05 1.1 1.15 1.2; do 
for window in 0 10 25 35 50; do 
for guidance_scale in 0.0 1.0 1.5; do 
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /home/sunshangquan/anaconda3/envs/p2p/bin/python evaluate_coco.py \
    --metrics "psnr" "ssim" "lpips" "mse" --tgt_path input --tgt_methods "/data/proj1/diffusion-inversion/res_paper/coco/"$model"_ReNoise_"$spacing"_4_"$window"_"$guidance_scale ;
done
done
done
done


# GNRI
for model in SDXL_Turbo; do 
for spacing in 0.8 0.85 0.9 0.95 1.0 1.05 1.1 1.15 1.2; do 
for window in 0 10 25 35 50; do 
for guidance_scale in 0.0 1.0 1.5; do 
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /home/sunshangquan/anaconda3/envs/p2p/bin/python evaluate_coco.py \
    --metrics "psnr" "ssim" "lpips" "mse" --tgt_path input --tgt_methods "/data/proj1/diffusion-inversion/res_paper/coco/"$model"_GNRI_"$spacing"_4_"$window"_"$guidance_scale ;
done
done
done
done

############################################################


# DDIM
for model in SDXL; do 
for spacing in 0.95 1.0 1.05; do 
for window in 0 5 8 10; do
for guidance_scale in 0.0 1.0 1.5; do 
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /home/sunshangquan/anaconda3/envs/p2p/bin/python evaluate_coco.py \
    --metrics "psnr" "ssim" "lpips" "mse" --tgt_path input --tgt_methods "/data/proj1/diffusion-inversion/res_paper/coco/"$model"_DDIM_"$spacing"_50_"$window"_"$guidance_scale ;
done
done
done
done


# NPI
for model in SDXL; do 
for spacing in 0.95 1.0 1.05; do 
for window in 0 5 8 10; do
for guidance_scale in 0.0 1.0 1.5; do 
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /home/sunshangquan/anaconda3/envs/p2p/bin/python evaluate_coco.py \
    --metrics "psnr" "ssim" "lpips" "mse" --tgt_path input --tgt_methods "/data/proj1/diffusion-inversion/res_paper/coco/"$model"_NPI_"$spacing"_50_"$window"_"$guidance_scale ;
done
done
done
done

# ReNoise
for model in SDXL; do 
for spacing in 0.95 1.0 1.05; do 
for window in 0 5 8 10; do
for guidance_scale in 0.0 1.0 1.5; do 
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /home/sunshangquan/anaconda3/envs/p2p/bin/python evaluate_coco.py \
    --metrics "psnr" "ssim" "lpips" "mse" --tgt_path input --tgt_methods "/data/proj1/diffusion-inversion/res_paper/coco/"$model"_ReNoise_"$spacing"_50_"$window"_"$guidance_scale ;
done
done
done
done


# GNRI
for model in SDXL; do 
for spacing in 0.95 1.0 1.05; do 
for window in 0 5 8 10; do
for guidance_scale in 0.0 1.0 1.5; do 
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /home/sunshangquan/anaconda3/envs/p2p/bin/python evaluate_coco.py \
    --metrics "psnr" "ssim" "lpips" "mse" --tgt_path input --tgt_methods "/data/proj1/diffusion-inversion/res_paper/coco/"$model"_GNRI_"$spacing"_50_"$window"_"$guidance_scale ;
done
done
done
done