
import torch
import numpy as np


class TRDI():
	def __init__(self, num_step=50, spacing=1.05, window=1):
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
		print(min_loss, best_steps, len(best_steps))
		# print([i-j for i, j in zip(best_steps[:-1], timesteps)])

		return best_steps[:self.num_step]
