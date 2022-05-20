import gym
import time
import os
import platform
from rl.finite_mdp import mdp
from rl.q_learning import q_learning


CLEAR_TERMINAL = 'cls' if platform.system() == 'Windows' else 'clear'
RENDER_FPS = 3

class TaxiV3CustomEnv:
	"""
		Custom taxi-v3 env to change rendering behavior, specifically to clear
		the terminal after each frame
	"""

	def __init__(self):
		self.env = gym.make("Taxi-v3")

	def __getattr__(self, attr):
		# get the same attributes as those of gym's environment
		return self.env.__getattribute__(attr)

	def step(self, action):
		self.iter += 1
		self.action = action
		self.obs, self.r, self.done, self.info = self.env.step(action)
		return self.obs, self.r, self.done, self.info

	def render(self):
		os.system(CLEAR_TERMINAL)
		self.env.render()
		print(f"iter {self.iter} : state {self.obs}, action {self.action}, reward {self.r}")
		time.sleep(1 / RENDER_FPS)

	def reset(self):
		self.obs = self.env.reset()
		self.iter = 0
		self.r, self.done, self.info, self.action = 0, False, None, None
		return self.obs


if __name__ == "__main__":
	GAMMA = 0.9
	ALPHA = 0.1
	EPS = 0.1

	env = TaxiV3CustomEnv()

	# Uncomment one of the following
	
	mdp(env, gamma=GAMMA)
	
	# q_learning(	
	# 	env,
	# 	n_actions=env.action_space.n,
	# 	gamma=GAMMA,
	# 	alpha=ALPHA,
	# 	eps=EPS,
	# 	max_frames=200,
	# 	render_frequency=1000
	# )