
conda activate diffusion-inversion

for model in SDXL_Turbo; do
    for spacing in 0.9 1.0 1.05; do
        for window in 10 25 50; do
            /home/sunshangquan/anaconda3/envs/diffusion-inversion/bin/python exp/unet_based/ddim.py --spacing $spacing --num_inference_steps 4 --model_type $model --trdi_window $window ; 
        done
    done
done