import flappy_bird_gym
import numpy as np
from gym import Env
import time


def q_init_state(n_actions:int) -> np.ndarray:
	"""
		Define how the value of each state-action pair for a newly found state
		should be initialized.

		- `n_actions`: size of action space

		Return array of default values whose size is that of the action space, each
		element corresponds to the value of each action.
	"""
	# Initially higher value would encourage exploration, but too high value would cause
	# the agent to never find an optimal policy
	return np.zeros(n_actions)

def q_epsilon_greedy(Q_table:dict, n_actions:int, state, eps:float) -> int:
	"""
		Epsilon-greedy policy that picks the current best action whose value is
		the highest in the current state with probability of `1 - eps`, and
		a random action in the action space with probability of `eps`.

		- `Q_table`: the dictionary with keys as states and values as array of values whose
			index is the action
		- `n_actions`: size of action space
		- `state`: the representation of current state. Must by immutable to be hashed as key of `Q_table`
		- `eps`: the number Epsilon, between 0 and 1. Set to 0 to disable exploration or 1 to
			take totally random actions

		Return the action
	"""
	if state not in Q_table.keys():
		Q_table[state] = q_init_state(n_actions)
	if np.random.random() < eps:
		# Exploration: choose a random action
		return np.random.randint(n_actions)
	# Exploitation: choose randomly among best actions (if there are many)
	return np.random.choice(np.arange(n_actions)[Q_table[state] == np.max(Q_table[state])])

def q_update(Q_table:dict, s0, a:int, s1, reward:float, gamma:float, alpha:float):
	"""
		Update the Q table.

		- `Q_table`: the dictionary with keys as states and values as array of values whose
			index is the action
		- `s0`: the representation of previous state. Must by immutable to be hashed as key of `Q_table`
		- `a`: the action taken in the previous state `s0`
		- `s1`: the representation of current state where the agent arrives by taking the action `a` in the 
			previous state `s0`. Must by immutable to be hashed as key of `Q_table`
	"""
	if s1 not in Q_table.keys():
		Q_table[s1] = q_init_state(len(Q_table[s0]))
	Q_table[s0][a] = Q_table[s0][a] + alpha * (reward + gamma * np.max(Q_table[s1]) - Q_table[s0][a])

def q_learning(env: Env, gamma:float, alpha:float, eps:float, transform_state_func=None, max_frames=100, render_frequency=10):
	"""
		Play out the episodes and update the Q table as the agent learn the optimal policy.

		- `env`: Gym environment with finite action space
		- `gamma`: discount factor for action value in future time steps (between 0 and 1)
		- `alpha`: learning rate (between 0 and 1). The bigger, the more impact newer values have over old values in 
			the Q table
		- `eps`: value for the epsilon-greedy policy (between 0 and 1)
		- `transform_state_func` (Optional): if given, the function transforms the observed state returned by the
			environment to a more adapted representation for the Q table. Example use: turning an unbounded continuous
			state space to discretized one to reduce the Q table's space; or turn numpy-array state representation to
			immutable Python tuple.
		- `max_frames`: maximum time steps the agent can take to finish an episode
		- `render_frequency`: number of episodes after which an episode is rendered for visualization

		After rendering an episode, user is prompted to enter a new value for `eps` (the discount factor), or
		enter `q` to quit the program. If input is empty or not a valid number, the learning continues with the current
		`eps`, otherwise the new `eps` is set. If `eps` is zero (no exploration), it is assumed that user would like to
		visualize the performance of the current policy, therefore every subsequent episode will be rendered. Otherwise
		if `eps` is not zero, the agent continues to learn for the next `render_frequency` episodes before the next
		visualization.
	"""
	Q_table = dict()

	i = 1
	while True:
		r = 0
		s = env.reset()
		if transform_state_func is not None:
			s = transform_state_func(env, s, 0)

		rendering = (i % render_frequency == 0) or (eps == 0)
		if rendering:
			print("Episode", i)

		t = 0
		while t < max_frames:
			action = q_epsilon_greedy(Q_table, env.action_space.n, s, eps)
			ss, reward, done, info = env.step(action)
			reward -= 1	# remove the reward for surviving, encouraging long term reward
			if done:
				reward -= 1000
			r += reward
			
			if rendering:
				env.render()
				time.sleep(1 / RENDER_FPS)

			if transform_state_func is not None:
				ss = transform_state_func(env, ss, info['score'])
			q_update(Q_table, s, action, ss, reward, gamma, alpha)
			if done:
				if rendering:
					print("FINISHED. Total reward:", r)
					user = input(f"Enter 'q' to quit or new epsilon for greedy policy (currently {eps}): ")
					if user == 'q':
						exit()
					try:
						eps = float(user)
					except:
						pass
				break
			s = ss
		i += 1
		t += not rendering
		alpha *= (1 - 1e-9)	# alpha decaying

def transform_state(env:Env, state:np.ndarray, score:int) -> tuple:
	return (score > 0, env._game.player_vel_y, ) + tuple((state // [15, 5]).astype(int))

if __name__ == "__main__":
	RENDER_FPS = 60

	GAMMA = 0.95
	ALPHA = 0.7
	EPS = 1e-32

	env = flappy_bird_gym.make('FlappyBird-v0')
	env._normalize_obs = False
	q_learning(env, gamma=GAMMA, alpha=ALPHA, eps=EPS, transform_state_func=transform_state, max_frames=1000, render_frequency=10000)