
import torch
import numpy as np


class TRDI():
	def __init__(self, num_step=50, spacing=1.05, window=1, verbose=False):
		beta_start = 0.00085
		beta_end = 0.012
		
		self.num_train_timesteps = 1000
		self.num_step = num_step

		self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, self.num_train_timesteps, dtype=torch.float32)
		self.delta_beta = self.betas[1] - self.betas[0]
		self.betas = self.betas ** 2
		self.alphas = 1 - self.betas
		self.alphas_cumprod = torch.cumprod(self.alphas, 0)
		self.ts = np.linspace(1,self.num_train_timesteps,self.num_train_timesteps)
		self.spacing = spacing
		self.window = window
		self.steps_offset = 1
		self.verbose = verbose
		
	def init_timesteps(self, timestep_spacing):
		if timestep_spacing == "linspace":
			timesteps = np.linspace(0, self.num_train_timesteps - 1, self.num_step).round()[::-1].copy().astype(np.int64)
		elif timestep_spacing == "leading":
			step_ratio = self.num_train_timesteps // self.num_step
			timesteps = (np.arange(0, self.num_step) * step_ratio).round()[::-1].copy().astype(np.int64)
			timesteps += self.steps_offset
		elif timestep_spacing == "trailing":
			step_ratio = self.num_train_timesteps / self.num_step
			timesteps = np.round(np.arange(self.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
			timesteps -= 1
		return timesteps

	def get_timesteps(self):
		lb = 0
		ub = self.num_train_timesteps
		steps = self.num_step + 1 
		span = (ub-lb)
		dx = 1.0 / (steps - 1)
		timesteps = [int(999 - ( lb + ( (i) * dx )**self.spacing*span) ) for i in range(steps)]
		if self.verbose:
			print(timesteps, len(timesteps), timesteps[:self.num_step])
		return timesteps[:self.num_step]
	
	
	def rescaling_timesteps(self, timesteps):
		ascending = timesteps[0] < timesteps[1]
		lb = min(timesteps)
		
        # ub = self.num_train_timesteps-1
		ub = max(timesteps)
		steps = len(timesteps) 
		span = (ub-lb)
		dx = 1.0 / (steps - 1)
		timesteps = [int( lb + ( (i) * dx )**self.spacing * span )  for i in range(steps)]
		if self.verbose:
			print(timesteps, len(timesteps), timesteps[:self.num_step])
		assert max(timesteps) <= self.num_train_timesteps
		assert min(timesteps) > 0
		if ascending:
			return timesteps
		else:
			return timesteps[::-1]
	
	def get_alpha(self, t):
		if t < 0:
			return self.alphas_cumprod[0]#(self.alphas_cumprod[0]**0.5 + self.delta_beta * t)**2
		elif t > self.num_train_timesteps-1:
			dt = t - self.num_train_timesteps + 1
			return self.alphas_cumprod[self.num_train_timesteps-1]#(self.alphas_cumprod[self.num_train_timesteps-1]**0.5 + self.delta_beta * dt)**2
		return self.alphas_cumprod[t]
	def compute_d(self, t, dt):
		if dt == 1:
			result = np.abs(2*self.compute_d(t, dt+1) - self.compute_d(t, dt+2))
		else:
			a1 = self.get_alpha(t)**0.5
			a2 = ((1/self.get_alpha(t-1)) - 1)**0.5
			a3 = ((1/self.get_alpha(t-dt)) - 1)**0.5
			result = (a1 * (a2 - a3)).abs()
		return result
	def reschedule(self, timesteps, ):
		window = self.window
		timesteps_sorted = sorted(timesteps)
		step_recoder = {}
		for i, step in enumerate(timesteps_sorted):
			step_recoder_step = {}
			left = max(1, step - window)
			right = min(self.num_train_timesteps, step + window+1)
			for t in range(left, right):
				if i == 0:
					loss = self.compute_d(t, t - 1)
					step_recoder_step[t] = [loss, [t,1]]
				else:
					min_loss = 1e10
					min_step = None
					for j, item in step_recoder[i-1].items():
						l, steps = item
						last_step = steps[0]
						if t - last_step < 2:
							continue
						loss_cur = l + self.compute_d(t, t - last_step)
						if loss_cur < min_loss:
							min_loss = loss_cur
							min_step = [t,]+steps
					step_recoder_step[t] = [min_loss, min_step]
			step_recoder[i] = step_recoder_step
		# print(step_recoder)
		min_loss = 1e10
		best_steps = None
		for i, item in step_recoder[len(timesteps)-1].items():
			loss, steps = item
			if loss < min_loss:
				best_steps = steps
				min_loss = loss
		if self.verbose:
			print(min_loss, best_steps, len(best_steps))
		# print([i-j for i, j in zip(best_steps[:-1], timesteps)])

		return best_steps[:self.num_step]


class AdaptiveTRDI(TRDI):
	"""Parameter-free analytic timestep allocation for TRDI probes.

	This keeps the original TRDI implementation untouched and builds a schedule
	by equalizing a cumulative error-density curve over the leading DDIM range.
	The density is computed only from the noise schedule, so it has no runtime
	overhead during inversion/sampling.
	"""

	def __init__(self, num_step=50, density_mode="balanced", verbose=False):
		super().__init__(num_step=num_step, spacing=1.0, window=0, verbose=verbose)
		self.density_mode = density_mode

	def _leading_bounds(self):
		timesteps = self.init_timesteps("leading")
		return int(min(timesteps)), int(max(timesteps))

	def _normalize(self, values):
		values = np.asarray(values, dtype=np.float64)
		values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
		values = np.maximum(values, 0.0)
		mean = values.mean()
		if mean <= 1e-12:
			return np.ones_like(values)
		return values / mean

	def _density(self, lb, ub):
		alpha = self.alphas_cumprod.detach().cpu().numpy().astype(np.float64)
		alpha = np.clip(alpha, 1e-8, 1.0 - 1e-8)
		sigma = np.sqrt((1.0 / alpha) - 1.0)
		logsnr = np.log(alpha) - np.log1p(-alpha)

		steps = np.arange(lb + 1, ub + 1)
		noise_delta = np.sqrt(alpha[steps]) * np.abs(sigma[steps] - sigma[steps - 1])
		logsnr_delta = np.abs(logsnr[steps] - logsnr[steps - 1])
		curvature = np.abs(logsnr[steps] - 2.0 * logsnr[steps - 1] + logsnr[np.maximum(steps - 2, 0)])

		noise_delta = self._normalize(noise_delta)
		logsnr_delta = self._normalize(logsnr_delta)
		curvature = self._normalize(curvature)

		if self.density_mode == "noise":
			density = noise_delta
		elif self.density_mode == "noise_floor25":
			density = 0.75 * noise_delta + 0.25
		elif self.density_mode == "noise_floor50":
			density = 0.50 * noise_delta + 0.50
		elif self.density_mode == "noise_floor75":
			density = 0.25 * noise_delta + 0.75
		elif self.density_mode == "logsnr":
			density = logsnr_delta
		elif self.density_mode == "curvature":
			density = curvature
		elif self.density_mode == "late":
			late_prior = ((ub - steps + 1) / max(ub - lb, 1)) ** 0.5
			density = 0.65 * noise_delta + 0.35 * self._normalize(late_prior)
		else:
			density = 0.50 * noise_delta + 0.35 * logsnr_delta + 0.15 * curvature

		return np.maximum(density, 1e-6)

	def _quantile_schedule(self, lb, ub, density):
		cum = np.concatenate([[0.0], np.cumsum(density)])
		total = float(cum[-1])
		targets = np.linspace(0.0, total, self.num_step)
		ascending = []
		for target in targets:
			index = int(np.searchsorted(cum, target, side="left"))
			ascending.append(lb + min(index, ub - lb))

		ascending[0] = lb
		ascending[-1] = ub
		min_gap = 2
		for i in range(1, len(ascending)):
			if ascending[i] < ascending[i - 1] + min_gap:
				ascending[i] = ascending[i - 1] + min_gap
		if ascending[-1] > ub:
			ascending[-1] = ub
			for i in range(len(ascending) - 2, -1, -1):
				if ascending[i] > ascending[i + 1] - min_gap:
					ascending[i] = ascending[i + 1] - min_gap
		if ascending[0] < lb:
			ascending[0] = lb
			for i in range(1, len(ascending)):
				if ascending[i] < ascending[i - 1] + min_gap:
					ascending[i] = ascending[i - 1] + min_gap
		ascending[0] = max(1, ascending[0])
		ascending[-1] = min(ub, ascending[-1])
		return [int(t) for t in ascending[::-1]]

	def get_timesteps_adaptive(self, bounds=None):
		if bounds is None:
			lb, ub = self._leading_bounds()
		else:
			lb, ub = bounds
			lb, ub = int(lb), int(ub)
			if lb >= ub:
				raise ValueError(f"invalid adaptive bounds: {bounds}")
		density = self._density(lb, ub)
		timesteps = self._quantile_schedule(lb, ub, density)
		if self.verbose:
			print(timesteps, len(timesteps), self.density_mode)
		return timesteps
