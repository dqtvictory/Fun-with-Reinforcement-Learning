import numpy as np
import time
from gym import Env
from dataclasses import dataclass


@dataclass
class QParams:
	n_actions: int
	alpha: float
	gamma: float
	eps: float
	default_action_value: float



def q_init_state(p:QParams) -> np.ndarray:
	"""
		Define how the value of each state-action pair for a newly found state
		should be initialized.

		- `n_actions`: size of action space
		- `default_value`: the default action-value Q(s,a)

		Return array of default values whose size is that of the action space, each
		element corresponds to the value of each action.
	"""
	# Initially higher value would encourage exploration, but too high value would cause
	# the agent to never find an optimal policy
	return np.ones(p.n_actions) * p.default_action_value

def q_epsilon_greedy(Q_table:dict, state, p:QParams) -> int:
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
		Q_table[state] = q_init_state(p)
	if np.random.random() < p.eps:
		# Exploration: choose a random action
		return np.random.randint(p.n_actions)
	# Exploitation: choose randomly among best actions (if there are many)
	return np.random.choice(np.arange(p.n_actions)[Q_table[state] == np.max(Q_table[state])])

def q_update(Q_table:dict, action:int, s0, s1, reward:float, p: QParams):
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
		Q_table[s1] = q_init_state(p)
	Q_table[s0][action] += p.alpha * (reward + p.gamma * np.max(Q_table[s1]) - Q_table[s0][action])

def q_learning(	env: Env,
				n_actions:int,
				gamma:float,
				alpha:float,
				eps:float,
				default_action_value=0,
				max_frames=100,
				render_frequency=10,
				render_fps=60,
			):
	"""
		Play out the episodes and update the Q table as the agent learn the optimal policy.

		- `env`: Gym environment with finite action space
		- `n_actions`: number of possible actions
		- `gamma`: discount factor for action value in future time steps (between 0 and 1)
		- `alpha`: learning rate (between 0 and 1). The bigger, the more impact newer values have over old values in 
			the Q table
		- `eps`: value for the epsilon-greedy policy (between 0 and 1)
		- `default_action_value`: value given to a new state-action pair when first seen by the agent
		- `max_frames`: maximum time steps the agent can take to finish an episode
		- `render_frequency`: number of episodes after which an episode is rendered for visualization
		- `render_fps`: frames per second for rendering

		After rendering an episode, user is prompted to enter a new value for `eps` (the discount factor), or
		enter `q` to quit the program. If input is empty or not a valid number, the learning continues with the current
		`eps`, otherwise the new `eps` is set. If `eps` is zero (no exploration), it is assumed that user would like to
		visualize the performance of the current policy, therefore every subsequent episode will be rendered. Otherwise
		if `eps` is not zero, the agent continues to learn for the next `render_frequency` episodes before the next
		visualization.
	"""
	Q_params = QParams(n_actions, alpha, gamma, eps, default_action_value)
	Q_table = dict()

	i = 1
	while True:
		r = 0
		s = env.reset()

		rendering = (i % render_frequency == 0) or (eps == 0)
		if rendering:
			print(f"Rendering episode {i} with zero-greedy policy")
			Q_params.eps = 0

		t = 0
		while t < max_frames:
			action = q_epsilon_greedy(Q_table, s, Q_params)
			ss, reward, done, _ = env.step(action)
			r += reward
			
			if rendering:
				env.render()
				time.sleep(1 / render_fps)

			q_update(Q_table, action, s, ss, reward, Q_params)
			
			if done:
				if rendering:
					print("  Total reward:", r)
					user = input(f"  Enter 'q' to quit or new epsilon (currently {eps}): ")
					if user == "q":
						return
					try:
						eps = float(user)
					except:
						pass
					Q_params.eps = eps
				break
			s = ss
			t += not rendering
		i += 1
