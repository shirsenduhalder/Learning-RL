import gym_bandits
import gym
import numpy as np
import sys, math

env = gym.make("BanditTenArmedGaussian-v0")
num_rounds = 20000
no_hands = env.action_space.n

count = np.zeros(no_hands)
rewards = np.zeros(no_hands)
Q = np.zeros(no_hands)

def epsilon_greedy(epsilon):
	rand = np.random.random()
	if rand < epsilon:
		action = env.action_space.sample()
	else:
		action = np.argmax(Q)

	return action

def softmax_policy(tau):
	total = np.sum(np.exp(Q/tau))
	probs = np.exp(Q/tau)/total

	threshold = np.random.random()
	cumulative_prob = 0.0

	for i in range(len(probs)):
		cumulative_prob += probs[i]
		if cumulative_prob > threshold:
			return i

	return np.argmax(probs)

def UCB_policy(iters, num_bandits, Q, count):
	ucb = np.zeros(num_bandits)

	if iters < 10:
		return i
	else:
		upper_bound = np.sqrt((2*np.log(np.sum(count)))/count)
		ucb = Q + upper_bound

	return np.argmax(ucb)

def thomson_sampling(alpha, beta, no_hands):
	samples = np.random.beta(alpha, beta)

	return np.argmax(samples)

env.reset()

#EPSILON GREEDY
for i in range(num_rounds):
	sys.stdout.write("\rRound {}/{}".format(i+1, num_rounds))
	sys.stdout.flush()
	arm = epsilon_greedy(0.5)

	observation, reward, done, _ = env.step(arm)

	count[arm] += 1
	rewards[arm] += reward
	Q[arm] = rewards[arm]/count[arm]

print("\nThe optimal arm using  Epsilon Greedy is: {}".format(np.argmax(Q)))


#SOFTMAX POLICY
for i in range(num_rounds):
	sys.stdout.write("\rRound {}/{}".format(i+1, num_rounds))
	sys.stdout.flush()
	arm = softmax_policy(0.5)

	observation, reward, done, _ = env.step(arm)

	count[arm] += 1
	rewards[arm] += reward
	Q[arm] = rewards[arm]/count[arm]

print("\nThe optimal arm using Softmax policy is: {}".format(np.argmax(Q)))

#UCB POLICY
for i in range(num_rounds):
	sys.stdout.write("\rRound {}/{}".format(i+1, num_rounds))
	sys.stdout.flush()
	arm = UCB_policy(i, no_hands, Q, count)

	observation, reward, done, _ = env.step(arm)

	count[arm] += 1
	rewards[arm] += reward
	Q[arm] = rewards[arm]/count[arm]

print("\nThe optimal arm using UCB policy is: {}".format(np.argmax(Q)))

#THOMPSON SAMPLING
alpha = np.ones(no_hands)
beta = np.ones(no_hands)

for i in range(num_rounds):
	sys.stdout.write("\rRound {}/{}".format(i+1, num_rounds))
	sys.stdout.flush()
	arm = thomson_sampling(alpha, beta, num_rounds)
	observation, reward, done, infor = env.step(arm)

	count[arm] += 1
	rewards[arm] += reward
	Q[arm] = rewards[arm]/count[arm]

	if reward > 0:
		alpha[arm] += 1
	else:
		beta[arm] += 1

print("\nThe optimal arm using Thompson Sampling is: {}".format(np.argmax(Q)))
