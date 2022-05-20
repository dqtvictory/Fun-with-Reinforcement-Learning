import numpy as np
from gym import Env	# for linter


def policy_iteration(env:Env, gamma:float, thresh=1e-3) -> dict:
	"""
		Policy iteration algorithm to approximate the optimal state value
		given that the problem is MDP-deterministic.
		
		- `env`: Gym environment with finite state and action space. There must be
			a dictionary stored at `env.env.P` that describe the fully deterministic
			Markov Decision Process (MDP) where `env.env.P[s][a]` is a list of destination
			states with associated reward and probability of reaching that state from
			`s` by taking action `a`
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
		
		- `env`: Gym environment with finite state and action space. There must be
			a dictionary stored at `env.env.P` that describe the fully deterministic
			Markov Decision Process (MDP) where `env.env.P[s][a]` is a list of destination
			states with associated reward and probability of reaching that state from
			`s` by taking action `a`
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

def mdp(env:Env, gamma:float, policy_optimizer="value_iteration", max_frames=100):
	"""
		Learn the optimal policy using one of the available algorithms
		then render each episode of the playthrough. When one ends, user
		is prompted to start a new episode or enter `q` to quit the program.

		- `env`: Gym environment with finite state and action space
		- `gamma`: discount factor for state value in future time steps (between 0 and 1)
		- `policy_optimizer`: algorithm to find optimal policy, being either `value_iteration`
			(recommended for faster convergence) or `policy_iteration`
		- `max_frames`: maximum frames to play for each episode
	"""
	if policy_optimizer == "value_iteration":
		policy = value_iteration(env, gamma)
	elif policy_optimizer == "policy_iteration":
		policy = policy_iteration(env, gamma)
	else:
		raise ValueError(f"No algorithm named '{policy_optimizer}'")

	while True:
		r = 0
		obs = env.reset()
		env.render()
		for _ in range(max_frames):
			action = policy[obs]
			obs, reward, done, _ = env.step(action)
			r += reward
			env.render()

			if done:
				print("FINISHED. Total reward:", r)
				user = input("  Enter 'q' to quit: ")
				if user == 'q':
					return
				else:
					break
