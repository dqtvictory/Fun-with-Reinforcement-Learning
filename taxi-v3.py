import gym
import time
import numpy as np
import os
import platform
from gym import Env	# for linter


################ Deterministic Markov Decision Process learning ################

def policy_iteration(env:Env, gamma:float, thresh=1e-3) -> dict:
	"""
		Policy iteration algorithm to approximate the optimal state value
		given that the problem is MDP-deterministic.
		
		- `env`: Gym environment with finite state and action space
		- `gamma`: discount factor for state value of future time steps (between 0 and 1)
		- `thresh`: small convergence threshold
		
		Returns the optimal policy.
	"""
	n_states, n_actions = env.observation_space.n, env.action_space.n
	V_old = np.zeros(n_states)
	policy = {s:0 for s in range(n_states)}

	while True:
		# Use current policy to update state value
		V_new = np.copy(V_old)
		for s in range(n_states):
			a = policy[s]	# pick action according to policy
			v = 0		# compute state value
			for p, ss, r, _ in env.env.P[s][a]:
				v += p * (r + gamma * V_old[ss])
			V_new[s] = v
		
		# Update policy
		for s in range(n_states):
			vs = -np.inf	# find best action whose subsequent state is maximized
			for a in range(n_actions):
				v = 0
				for p, ss, r, _ in env.env.P[s][a]:
					v += p * (r + gamma * V_new[ss])
				if v > vs:
					policy[s] = a
					vs = v

		# Check for convergence
		if np.all(np.abs(V_old - V_new) <= thresh):
			break
		V_old = V_new
	return policy

def value_iteration(env:Env, gamma:float, thresh=1e-3) -> dict:
	"""
		Value iteration algorithm to approximate the optimal state value
		given that the problem is MDP-deterministic. This algorithm is about
		20% faster than policy iteration because it iterates the state space
		and updates the state value in only one go.
		
		- `env`: Gym environment with finite state and action space
		- `gamma`: discount factor for state value of future time steps (between 0 and 1)
		- `thresh`: small convergence threshold
		
		Returns the optimal policy.
	"""
	n_states, n_actions = env.observation_space.n, env.action_space.n
	V_old = np.zeros(n_states)
	policy = {s:0 for s in range(n_states)}

	while True:
		# Iterate over the states and update policy and state value directly
		V_new = np.copy(V_old)
		for s in range(n_states):
			vs = -np.inf	# compute state value based on value of state reached by making best action
			for a in range(n_actions):
				v = 0
				for p, ss, r, _ in env.env.P[s][a]:
					v += p * (r + gamma * V_old[ss])
				if v > vs:
					policy[s] = a	# update policy
					vs = v
			V_new[s] = vs
		
		# Check for convergence
		if np.all(np.abs(V_old - V_new) <= thresh):
			break
		V_old = V_new
	return policy

def taxi_mdp(env:Env, gamma:float):
	"""
		Learn the optimal policy using one of the available algorithms
		then render each episode of the playthrough. When one ends, user
		is prompted to start a new episode or enter `q` to quit the program.

		- `env`: Gym environment with finite state and action space
		- `gamma`: discount factor for state value in future time steps (between 0 and 1)
	"""
	env = gym.make('Taxi-v3')
	policy = MDP_LEARNING_FUNC(env, gamma)

	while True:
		r = 0
		obs = env.reset()
		for i in range(100):
			action = policy[obs]
			obs, reward, done, _ = env.step(action)
			r += reward

			os.system(CLEAR_TERMINAL)
			env.render()
			print(f"iter {i} : state {obs}, action {action}, reward {reward}")

			time.sleep(1 / RENDER_FPS)
			if done:
				print("\FINISHED. Total reward:", r)
				user = input("Enter 'q' to quit: ")
				if user == 'q':
					exit()
				else:
					break


################ Q-learning ################

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
	return np.ones(n_actions) * 10

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

def taxi_q_learning(env: Env, gamma:float, alpha:float, eps:float, transform_state_func=None, max_frames=100, render_frequency=10):
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
			s = transform_state_func(s)

		rendering = (i % render_frequency == 0) or (eps == 0)

		for _ in range(max_frames):
			if rendering:
				os.system(CLEAR_TERMINAL)
				print("Episode", i)
				env.render()
				time.sleep(1 / RENDER_FPS)

			action = q_epsilon_greedy(Q_table, env.action_space.n, s, eps)
			ss, reward, done, _ = env.step(action)
			r += reward
			if transform_state_func is not None:
				ss = transform_state_func(ss)
			q_update(Q_table, s, action, ss, reward, gamma, alpha)
			if done:
				if rendering:
					print("\FINISHED. Total reward:", r)
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


if __name__ == "__main__":
	CLEAR_TERMINAL = 'cls' if platform.system() == 'Windows' else 'clear'
	RENDER_FPS = 3

	GAMMA = 0.9
	ALPHA = 0.1
	EPS = 0.1

	MDP_LEARNING_FUNC = value_iteration

	env = gym.make('Taxi-v3')

	# Uncomment one of the following
	# taxi_mdp(env, gamma=GAMMA)
	taxi_q_learning(env, gamma=GAMMA, alpha=ALPHA, eps=EPS, max_frames=200, render_frequency=1000)