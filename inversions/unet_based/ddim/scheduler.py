import torch
from diffusers import DDIMScheduler
from typing import List, Optional, Tuple, Union
import numpy as np

class CustomInversionScheduler:
    pass


class CustomDDIMInversionScheduler(DDIMScheduler, CustomInversionScheduler):
    def ddim_inverse_step(self, noise_pred: torch.Tensor, timestep: int, sample: torch.Tensor):
        """inversion process based on the reversible assumption of ODE processes: z_{t-1} -> z_t

        Args:
            noise_pred (torch.Tensor): noise prediction from the diffusion UNet model
            timestep (int): current timestep t-1
            sample (torch.Tensor): latent code z_{t-1}

        Returns:
            torch.Tensor: latent code at next timestep z_t
        """
        prev_timestep = timestep
        timestep = min(
            timestep - self.config.num_train_timesteps // self.num_inference_steps, self.config.num_train_timesteps - 1
        )
        alpha_prod_t = self.alphas_cumprod[timestep] if timestep >= 0 else self.final_alpha_cumprod
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep]
        beta_prod_t = 1 - alpha_prod_t

        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
            pred_epsilon = noise_pred
        elif self.config.prediction_type == "sample":
            pred_original_sample = noise_pred
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * noise_pred
            pred_epsilon = (alpha_prod_t**0.5) * noise_pred + (beta_prod_t**0.5) * sample
        
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        return prev_sample
    
    def set_timesteps(self, num_inference_steps: int = None, device: Union[str, torch.device] = None, timesteps: List = None, ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
        """
        if timesteps is None:
            if num_inference_steps > self.config.num_train_timesteps:
                raise ValueError(
                    f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                    f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                    f" maximal {self.config.num_train_timesteps} timesteps."
                )

            self.num_inference_steps = num_inference_steps

            # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
            if self.config.timestep_spacing == "linspace":
                timesteps = (
                    np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps)
                    .round()[::-1]
                    .copy()
                    .astype(np.int64)
                )
            elif self.config.timestep_spacing == "leading":
                step_ratio = self.config.num_train_timesteps // self.num_inference_steps
                # creates integer timesteps by multiplying by ratio
                # casting to int to avoid issues when num_inference_step is power of 3
                timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
                timesteps += self.config.steps_offset
            elif self.config.timestep_spacing == "trailing":
                step_ratio = self.config.num_train_timesteps / self.num_inference_steps
                # creates integer timesteps by multiplying by ratio
                # casting to int to avoid issues when num_inference_step is power of 3
                timesteps = np.round(np.arange(self.config.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
                timesteps -= 1
            else:
                raise ValueError(
                    f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'leading' or 'trailing'."
                )

            self.timesteps = torch.from_numpy(timesteps).to(device)
        else:

            self.num_inference_steps = len(timesteps)
            self.timesteps = torch.Tensor(timesteps).to(torch.int32).to(device)

