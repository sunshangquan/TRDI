
conda activate diffusion-inversion

for model in SDXL; do
    for spacing in 1.05; do
        for window in 0; do
            /home/sunshangquan/anaconda3/envs/diffusion-inversion/bin/python exp/unet_based/gnri.py \
            --spacing $spacing --num_inference_steps 10 --model_type $model --trdi_window $window \
            # --path_name res__; 
        done
    done
done


for model in SDXL_Turbo; do
    for spacing in 0.8 0.85 0.9 0.95 1.0 1.05 1.1 1.15 1.2; do 
        for window in 0 10 25 50; do
            for guidance_scale in 0.0 1.0 2.0; do
            CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /home/sunshangquan/anaconda3/envs/diffusion-inversion/bin/python exp/unet_based/gnri.py \
            --spacing $spacing --num_inference_steps 4 --model_type $model --trdi_window $window \
            --guidance_scale $guidance_scale
            # --path_name res__; 
            done
        done
    done
done

for model in SD15; do
    for spacing in 1 1.05; do
        for window in 0 8; do
            for guidance_scale in 1.0; do
            CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /home/sunshangquan/anaconda3/envs/diffusion-inversion/bin/python exp/unet_based/gnri.py \
            --spacing $spacing --num_inference_steps 50 --model_type $model --trdi_window $window \
            --guidance_scale $guidance_scale; 
            done
        done
    done
done