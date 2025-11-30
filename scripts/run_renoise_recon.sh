
conda activate diffusion-inversion


for model in SD15; do
    for spacing in 1.0; do
        for window in 0 10; do
            /home/sunshangquan/anaconda3/envs/diffusion-inversion/bin/python exp/unet_based/renoise_recon.py \
            --spacing $spacing --num_inference_steps 50 --renoise_steps 1 --model_type $model --trdi_window $window \
            --early_weights 0.5 0.5 --latter_weights 0.0 1.0 \
            --guidance_scale 2.0; 
        done
    done
done


for model in SDXL; do
    for spacing in 0.95 1.0 1.05; do
        for window in 0 5 8 10; do
            for guidance_scale in 0.0 1.0 1.5; do
                CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /home/sunshangquan/anaconda3/envs/diffusion-inversion/bin/python exp/unet_based/renoise_recon.py \
                --spacing $spacing --num_inference_steps 50 --renoise_steps 1 --model_type $model --trdi_window $window \
                --early_weights 0.5 0.5 --latter_weights 0.0 1.0 \
                    --guidance_scale $guidance_scale; 
            done
        done
    done
done


for model in SDXL_Turbo; do
    for spacing in 0.8 0.85 0.9 0.95 1.0 1.05 1.1 1.15 1.2; do 
        for window in 0 10 25 35 50; do
            for guidance_scale in 0.0 1.0 1.5; do
                /home/sunshangquan/anaconda3/envs/diffusion-inversion/bin/python exp/unet_based/renoise_recon.py \
                --spacing $spacing --num_inference_steps 4 --renoise_steps 9 --model_type $model --trdi_window $window \
                --early_weights 0.2 0.2 0.2 0.2 0.2 0.0 0.0 0.0 0.0 0.0 --latter_weights 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.5 --guidance_scale $guidance_scale; 
            done
        done
    done
done