
conda activate diffusion-inversion

for model in SDXL_Turbo; do
    for spacing in 0.8 0.85 0.9 0.95 1.0; do #1.05 1.1 1.15 1.2
        for window in 0 10 25 50; do
            for guidance_scale in 1.0 2.0 7.5; do
            /home/sunshangquan/anaconda3/envs/diffusion-inversion/bin/python exp/unet_based/renoise.py \
            --spacing $spacing --num_inference_steps 4 --renoise_steps 9 --model_type $model --trdi_window $window \
            --early_weights 0.2 0.2 0.2 0.2 0.2 0.0 0.0 0.0 0.0 0.0 --latter_weights 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.5 --guidance_scale $guidance_scale; 
            done
        done
    done
done

# for model in SDXL_Turbo; do 
# for spacing in 0.9; do 
# for window in 10; do 
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /home/sunshangquan/anaconda3/envs/p2p/bin/python /home/sunshangquan/PnPInversion-main/evaluation/evaluate_my.py --metrics "structure_distance" "psnr_unedit_part" "lpips_unedit_part" "mse_unedit_part" "ssim_unedit_part" "clip_similarity_source_image" "clip_similarity_target_image" "clip_similarity_target_image_edit_part" \
# --result_path evaluation_result --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
# --tgt_path /data/PIE-Bench/results_  --annotation_images \
# --tgt_methods "/data/proj1/diffusion-inversion/res_paper/pie-bench/"$model"_ReNoise_"$spacing"_4_"$window"_0.0" ;
# done
# done
# done

for model in SDXL; do
    for spacing in 1.05; do
        for window in 8; do
            for guidance_scale in 2.0; do
            /home/sunshangquan/anaconda3/envs/diffusion-inversion/bin/python exp/unet_based/renoise.py \
            --spacing $spacing --num_inference_steps 50 --renoise_steps 1 --model_type $model --trdi_window $window \
            --early_weights 0.5 0.5 --latter_weights 0.0 1.0 --guidance_scale $guidance_scale; 
            done
        done
    done
done

for model in SD15; do
    for spacing in 1 1.05; do
        for window in 0 8; do
            for guidance_scale in 1.0; do
            CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /home/sunshangquan/anaconda3/envs/diffusion-inversion/bin/python exp/unet_based/renoise.py \
            --spacing $spacing --num_inference_steps 50 --renoise_steps 1 --model_type $model --trdi_window $window \
            --early_weights 0.5 0.5 --latter_weights 0.0 1.0 --guidance_scale $guidance_scale; 
            done
        done
    done
done