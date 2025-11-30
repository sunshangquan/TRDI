CUDA_VISIBLE_DEVICES=3
for model in SD15; do 
for spacing in 1.0 1.05 ; do 
for window in 0 8; do 
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /home/sunshangquan/anaconda3/envs/p2p/bin/python /home/sunshangquan/PnPInversion-main/evaluation/evaluate_my.py --metrics "structure_distance" "psnr_unedit_part" "lpips_unedit_part" "mse_unedit_part" "ssim_unedit_part" "clip_similarity_source_image" "clip_similarity_target_image" "clip_similarity_target_image_edit_part" \
--result_path evaluation_result --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
--tgt_path /data/PIE-Bench/results_  --annotation_images \
--tgt_methods "/data/proj1/diffusion-inversion/res/pie-bench/"$model"_DDIM_"$spacing"_50_"$window ;
done
done
done

CUDA_VISIBLE_DEVICES=0
for model in SDXL_Turbo; do
for spacing in 0.9 1.0 1.05 1.1 1.2 1.3; do
for window in 0 1 10 25 50; do
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /home/sunshangquan/anaconda3/envs/p2p/bin/python /home/sunshangquan/PnPInversion-main/evaluation/evaluate_my.py --metrics "structure_distance" "psnr_unedit_part" "lpips_unedit_part" "mse_unedit_part" "ssim_unedit_part" "clip_similarity_source_image" "clip_similarity_target_image" "clip_similarity_target_image_edit_part" \
--result_path evaluation_result --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
--tgt_path /data/PIE-Bench/results_  --annotation_images \
--tgt_methods "/data/proj1/diffusion-inversion/res/pie-bench/"$model"_DDIM_"$spacing"_4_"$window ;
done
done
done

CUDA_VISIBLE_DEVICES=1
for model in SD15; do 
for spacing in 0.9 ; do 
for window in 0 1 4; do 
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /home/sunshangquan/anaconda3/envs/p2p/bin/python /home/sunshangquan/PnPInversion-main/evaluation/evaluate_my.py --metrics "structure_distance" "psnr_unedit_part" "lpips_unedit_part" "mse_unedit_part" "ssim_unedit_part" "clip_similarity_source_image" "clip_similarity_target_image" "clip_similarity_target_image_edit_part" \
--result_path evaluation_result --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
--tgt_path /data/PIE-Bench/results_  --annotation_images \
--tgt_methods "/data/proj1/diffusion-inversion/res/pie-bench/"$model"_DDIM_"$spacing"_50_"$window ;
done
done
done

CUDA_VISIBLE_DEVICES=3
for model in SDXL; do 
for spacing in 1.1 0.9; do 
for window in 0; do 
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /home/sunshangquan/anaconda3/envs/p2p/bin/python /home/sunshangquan/PnPInversion-main/evaluation/evaluate_my.py --metrics "structure_distance" "psnr_unedit_part" "lpips_unedit_part" "mse_unedit_part" "ssim_unedit_part" "clip_similarity_source_image" "clip_similarity_target_image" "clip_similarity_target_image_edit_part" \
--result_path evaluation_result --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
--tgt_path /data/PIE-Bench/results_  --annotation_images \
--tgt_methods "/data/proj1/diffusion-inversion/res_paper/pie-bench/"$model"_DDIM_"$spacing"_50_"$window ;
done
done
done

# GNRI
for model in SDXL; do 
for spacing in 1.05; do 
for window in 0 10; do 
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /home/sunshangquan/anaconda3/envs/p2p/bin/python /home/sunshangquan/PnPInversion-main/evaluation/evaluate_my.py --metrics "structure_distance" "psnr_unedit_part" "lpips_unedit_part" "mse_unedit_part" "ssim_unedit_part" "clip_similarity_source_image" "clip_similarity_target_image" "clip_similarity_target_image_edit_part" \
--result_path evaluation_result --edit_category_list 0 \
--tgt_path /data/PIE-Bench/results_  --annotation_images \
--tgt_methods "/data/proj1/diffusion-inversion/res_paper/pie-bench/"$model"_GNRI_"$spacing"_10_"$window ;
done
done
done

# NPI
for model in SDXL; do 
for spacing in 1.0; do 
for window in 0; do 
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /home/sunshangquan/anaconda3/envs/p2p/bin/python /home/sunshangquan/PnPInversion-main/evaluation/evaluate_my.py --metrics "structure_distance" "psnr_unedit_part" "lpips_unedit_part" "mse_unedit_part" "ssim_unedit_part" "clip_similarity_source_image" "clip_similarity_target_image" "clip_similarity_target_image_edit_part" \
--result_path evaluation_result --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
--tgt_path /data/PIE-Bench/results_  --annotation_images \
--tgt_methods "/data/proj1/diffusion-inversion/res_paper/pie-bench/"$model"_NPI_"$spacing"_50_"$window ;
done
done
done

for model in SDXL_Turbo; do 
for spacing in 1.05; do 
for window in 25; do 
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /home/sunshangquan/anaconda3/envs/p2p/bin/python /home/sunshangquan/PnPInversion-main/evaluation/evaluate_my.py --metrics "structure_distance" "psnr_unedit_part" "lpips_unedit_part" "mse_unedit_part" "ssim_unedit_part" "clip_similarity_source_image" "clip_similarity_target_image" "clip_similarity_target_image_edit_part" \
--result_path evaluation_result --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
--tgt_path /data/PIE-Bench/results_  --annotation_images \
--tgt_methods "/data/proj1/diffusion-inversion/res_paper/pie-bench/"$model"_NPI_"$spacing"_4_"$window ;
done
done
done


# ReNoise

for model in SDXL_Turbo; do 
for spacing in 0.9 0.95 1.0 1.05 1.1 1.15 1.2; do 
for window in 0 10 25 50; do 
for guidance_scale in 1.0 2.0 7.5; do
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /home/sunshangquan/anaconda3/envs/p2p/bin/python /home/sunshangquan/PnPInversion-main/evaluation/evaluate_my.py --metrics "structure_distance" "psnr_unedit_part" "lpips_unedit_part" "mse_unedit_part" "ssim_unedit_part" "clip_similarity_source_image" "clip_similarity_target_image" "clip_similarity_target_image_edit_part" \
--result_path evaluation_result --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
--tgt_path /data/PIE-Bench/results_  --annotation_images \
--tgt_methods "/data/proj1/diffusion-inversion/res_paper/pie-bench/"$model"_ReNoise_"$spacing"_4_"$window"_"$guidance_scale ;
done
done
done
done

for model in SDXL; do 
for spacing in 1.05; do 
for window in 8; do 
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /home/sunshangquan/anaconda3/envs/p2p/bin/python /home/sunshangquan/PnPInversion-main/evaluation/evaluate_my.py --metrics "structure_distance" "psnr_unedit_part" "lpips_unedit_part" "mse_unedit_part" "ssim_unedit_part" "clip_similarity_source_image" "clip_similarity_target_image" "clip_similarity_target_image_edit_part" \
--result_path evaluation_result --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
--tgt_path /data/PIE-Bench/results_  --annotation_images \
--tgt_methods "/data/proj1/diffusion-inversion/res_paper/pie-bench/"$model"_ReNoise_"$spacing"_50_"$window"_2.0"  ;
done
done
done


for model in SD15; do 
for spacing in 1.0  1.05; do 
for window in 0 8; do 
for guidance_scale in 0.8 1.0 1.2; do
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /home/sunshangquan/anaconda3/envs/p2p/bin/python /home/sunshangquan/PnPInversion-main/evaluation/evaluate_my.py --metrics "structure_distance" "psnr_unedit_part" "lpips_unedit_part" "mse_unedit_part" "ssim_unedit_part" "clip_similarity_source_image" "clip_similarity_target_image" "clip_similarity_target_image_edit_part" \
--result_path evaluation_result --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
--tgt_path /data/PIE-Bench/results_  --annotation_images \
--tgt_methods "/data/proj1/diffusion-inversion/res_paper/pie-bench/"$model"_DDIM_"$spacing"_50_"$window"_"$guidance_scale  ;

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /home/sunshangquan/anaconda3/envs/p2p/bin/python /home/sunshangquan/PnPInversion-main/evaluation/evaluate_my.py --metrics "structure_distance" "psnr_unedit_part" "lpips_unedit_part" "mse_unedit_part" "ssim_unedit_part" "clip_similarity_source_image" "clip_similarity_target_image" "clip_similarity_target_image_edit_part" \
--result_path evaluation_result --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
--tgt_path /data/PIE-Bench/results_  --annotation_images \
--tgt_methods "/data/proj1/diffusion-inversion/res_paper/pie-bench/"$model"_ReNoise_"$spacing"_50_"$window"_"$guidance_scale  ;

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /home/sunshangquan/anaconda3/envs/p2p/bin/python /home/sunshangquan/PnPInversion-main/evaluation/evaluate_my.py --metrics "structure_distance" "psnr_unedit_part" "lpips_unedit_part" "mse_unedit_part" "ssim_unedit_part" "clip_similarity_source_image" "clip_similarity_target_image" "clip_similarity_target_image_edit_part" \
--result_path evaluation_result --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
--tgt_path /data/PIE-Bench/results_  --annotation_images \
--tgt_methods "/data/proj1/diffusion-inversion/res_paper/pie-bench/"$model"_NPI_"$spacing"_50_"$window"_"$guidance_scale  ;

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /home/sunshangquan/anaconda3/envs/p2p/bin/python /home/sunshangquan/PnPInversion-main/evaluation/evaluate_my.py --metrics "structure_distance" "psnr_unedit_part" "lpips_unedit_part" "mse_unedit_part" "ssim_unedit_part" "clip_similarity_source_image" "clip_similarity_target_image" "clip_similarity_target_image_edit_part" \
--result_path evaluation_result --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
--tgt_path /data/PIE-Bench/results_  --annotation_images \
--tgt_methods "/data/proj1/diffusion-inversion/res_paper/pie-bench/"$model"_GNRI_"$spacing"_50_"$window"_"$guidance_scale  ;
done
done
done
done