import gym
import random
env = gym.make('Taxi-v2')

env.render()
alpha = 0.4
gamma = 0.999
epsilon = 0.017

q = {}
for s in range(env.observation_space.n):
	for a in range(env.action_space.n):
		q[(s, a)] = 0.0

def update_Q_table(prev_state, action, reward, next_state, alpha, gamma):
	q_max = max([q[(next_state, act)] for act in range(env.action_space.n)])
	q[(prev_state, action)] += alpha*(reward  + gamma*q_max - q[(prev_state, action)])

def epsilon_greedy_policy(state, epsilon):
	if random.uniform(0, 1) < epsilon:
		return env.action_space.sample()
	else:
		return max(list(range(env.action_space.n)), key=lambda x:q[(state, x)])

num_episodes = 8000

for i in range(num_episodes):
	r = 0
	prev_state = env.reset()

	while True:
		env.render()
		action = epsilon_greedy_policy(prev_state, epsilon)
		next_state, reward, done, _ = env.step(action)
		update_Q_table(prev_state, action, reward, next_state, alpha, gamma)
		prev_state = next_state
		r+=reward
		if done:
			break
	print("Total Reward: {}".format(r))

env.close()