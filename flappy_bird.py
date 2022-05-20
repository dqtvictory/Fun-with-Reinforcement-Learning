import flappy_bird_gym
import numpy as np
from rl.q_learning import q_learning


class FlappyBirdCustomEnv():
	"""
		Custom flappy bird env to add more information to state observation and change
		default reward mechanism
	"""
	
	def __init__(self):
		self.env = flappy_bird_gym.make('FlappyBird-v0')
		self.env._normalize_obs = False			# disable normalization for state representation
		self.env.observation_space.shape = (4,)	# add 2 more dimensions to state representation

	def __getattr__(self, attr):
		# get the same attributes as those of gym's environment
		return self.env.__getattribute__(attr)

	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		obs = self.transform_state(obs, info['score'])	# add information to default state output
		reward -= 1		# don't reward just for surviving in the next frame
		if done:
			reward -= 1000	# punish when the bird dies
		return obs, reward, done, info

	def reset(self):
		return self.transform_state(self.env.reset(), 0)

	def transform_state(self, state:np.ndarray, score:int) -> tuple:
		# State: (starting game?, y speed, x distance to next pipe, y distance to pipe's open space)
		return (score > 0, self.env._game.player_vel_y, ) + tuple((state // [15, 5]).astype(int))


GAMMA = 0.95
ALPHA = 0.7
EPS = 1e-32

RENDER_FPS = 100

if __name__ == "__main__":
	env = FlappyBirdCustomEnv()
	q_learning(
		env,
		n_actions=env.action_space.n,
		gamma=GAMMA,
		alpha=ALPHA,
		eps=EPS,
		max_frames=1000000,
		render_frequency=10000,
		render_fps=RENDER_FPS
	)