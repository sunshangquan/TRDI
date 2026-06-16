from typing import List, Optional, Union

import numpy as np
import torch
from diffusers import DPMSolverMultistepInverseScheduler


class CustomDPMSolverMultistepInverseScheduler(DPMSolverMultistepInverseScheduler):
    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[List[int]] = None,
    ):
        if num_inference_steps is None and timesteps is None:
            raise ValueError("Must pass exactly one of `num_inference_steps` or `timesteps`.")
        if num_inference_steps is not None and timesteps is not None:
            raise ValueError("Can only pass one of `num_inference_steps` or `timesteps`.")
        if timesteps is not None and self.config.use_karras_sigmas:
            raise ValueError("Cannot use `timesteps` with `config.use_karras_sigmas = True`.")
        if timesteps is not None and self.config.use_exponential_sigmas:
            raise ValueError("Cannot use `timesteps` with `config.use_exponential_sigmas = True`.")
        if timesteps is not None and self.config.use_beta_sigmas:
            raise ValueError("Cannot use `timesteps` with `config.use_beta_sigmas = True`.")

        clipped_idx = torch.searchsorted(torch.flip(self.lambda_t, [0]), self.config.lambda_min_clipped).item()
        self.noisiest_timestep = self.config.num_train_timesteps - 1 - clipped_idx

        if timesteps is not None:
            timesteps = np.array(timesteps, dtype=np.int64)
        elif self.config.timestep_spacing == "linspace":
            timesteps = (
                np.linspace(0, self.noisiest_timestep, num_inference_steps + 1).round()[:-1].copy().astype(np.int64)
            )
        elif self.config.timestep_spacing == "leading":
            step_ratio = (self.noisiest_timestep + 1) // (num_inference_steps + 1)
            timesteps = (np.arange(0, num_inference_steps + 1) * step_ratio).round()[:-1].copy().astype(np.int64)
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / num_inference_steps
            timesteps = np.arange(self.noisiest_timestep + 1, 0, -step_ratio).round()[::-1].copy().astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.config.timestep_spacing} is not supported. Please choose 'linspace', 'leading', or 'trailing'."
            )

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        log_sigmas = np.log(sigmas)

        if self.config.use_karras_sigmas:
            sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas]).round().astype(np.int64)
            sigmas = np.concatenate([sigmas, sigmas[-1:]]).astype(np.float32)
        elif self.config.use_exponential_sigmas:
            sigmas = self._convert_to_exponential(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])
        elif self.config.use_beta_sigmas:
            sigmas = self._convert_to_beta(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])
        else:
            sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
            sigma_max = (
                (1 - self.alphas_cumprod[self.noisiest_timestep]) / self.alphas_cumprod[self.noisiest_timestep]
            ) ** 0.5
            sigmas = np.concatenate([sigmas, [sigma_max]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas)

        _, unique_indices = np.unique(timesteps, return_index=True)
        timesteps = timesteps[np.sort(unique_indices)]

        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)
        self.num_inference_steps = len(timesteps)
        self.model_outputs = [None] * self.config.solver_order
        self.lower_order_nums = 0
        self._step_index = None
        self.sigmas = self.sigmas.to("cpu")

