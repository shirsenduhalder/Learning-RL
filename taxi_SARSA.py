import gym
import sys
import random
env = gym.make('Taxi-v2')

env.render()
alpha = 0.85
gamma = 0.90
epsilon = 0.8

q = {}
for s in range(env.observation_space.n):
	for a in range(env.action_space.n):
		q[(s, a)] = 0.0

def epsilon_greedy_policy(state, epsilon):
	if random.uniform(0, 1) < epsilon:
		return env.action_space.sample()
	else:
		return max(list(range(env.action_space.n)), key=lambda x:q[(state, x)])

num_episodes = 8000

for i in range(num_episodes):
	r = 0
	sys.stdout.write('\rEpsiode: {}/{}'.format(i+1, num_episodes))
	sys.stdout.flush()
	state = env.reset()
	action = epsilon_greedy_policy(state, epsilon)
	while True:
		next_state, reward, done, _ = env.step(action)
		next_action = epsilon_greedy_policy(next_state, epsilon)
		
		q[(state, action)] += alpha*(reward + gamma*q[(next_state, next_action)] - q[(state, action)])

		action = next_action
		state = next_state
		r+=reward
		if done:
			break
	# print("Total Reward: {}".format(r))

env.close()