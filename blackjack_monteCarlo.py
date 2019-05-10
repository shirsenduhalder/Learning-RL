import numpy as np
import gym
import os, sys, argparse
from matplotlib import pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from functools import partial
plt.style.use('ggplot')

def arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('-t', "--type_visit", help="Enter type of visit. Options are First_Visit and Every_Visit", type=str, default="First_Visit")
	parser.add_argument('-s', "--save_dir", help="Save directory", required=True)

	return parser

def sample_policy(observation):
	score, dealer_score, usable_ace = observation

	return 0 if score >= 20 else 1

def generate_episode(policy, env):
	states, actions, rewards = [], [], []
	observation = env.reset()

	while True:
		states.append(observation)
		action = sample_policy(observation)
		actions.append(action)
		observation, reward, done, _ = env.step(action)
		rewards.append(reward)

		if done:
			break

	return states, actions, rewards

def mc_prediction(policy, env, n_episodes, type_visit='first_visit'):
	value_table = defaultdict(float)
	N = defaultdict(int)

	for episode in range(n_episodes):
		states, actions, rewards = generate_episode(policy, env)
		returns = 0
		sys.stdout.write("\rEpisode {}/{}".format(episode + 1, n_episodes))
		sys.stdout.flush()
		for t in range(len(states) - 1, -1, -1):
			R = rewards[t]
			S = states[t]

			returns += R

			if type_visit == 'First_Visit':
				if S not in states[:t]:
					N[S] += 1
					value_table[S] += (returns - value_table[S])/N[S]
			else:
				N[S] += 1
				value_table[S] += (returns - value_table[S])/N[S]

	return value_table


def plot_blackjack(values, type_visit, ax1, ax2):
	player_sum = np.arange(12, 21 + 1)
	dealer_show = np.arange(1, 10 + 1)
	usable_ace = np.array([False, True])

	state_values = np.zeros((len(player_sum), len(dealer_show), len(usable_ace)))

	for i, player in enumerate(player_sum):
		for j, dealer in enumerate(dealer_show):
			for k, ace in enumerate(usable_ace):
				state_values[i, j, k] = values[player, dealer, ace]

	X, Y = np.meshgrid(player_sum, dealer_show)

	ax1.plot_wireframe(X, Y, state_values[:, :, 0])
	ax2.plot_wireframe(X, Y, state_values[:, :, 1])

	for ax in ax1, ax2:
		ax.set_zlim(-1, 1)
		ax.set_ylabel("Player Sum")
		ax.set_xlabel("Dealer showing")
		ax.set_zlabel("State Value")


args = arguments()
args = args.parse_args()
env = gym.make('Blackjack-v0')
type_visit = args.type_visit
save_dir = args.save_dir

assert type_visit == "First_Visit" or type_visit == "Every_Visit", "Wrong Visit Type mentioned"
values = mc_prediction(sample_policy, env, 500000, type_visit)
fig, axes = plt.subplots(nrows=2, figsize=(5, 8), subplot_kw={'projection':'3d'})
fig.suptitle("{} Monte Carlo in Blackjack".format(type_visit.replace('_', " ")))
axes[0].set_title("Value function without usable ace")
axes[1].set_title("Value function with usable ace")
plot_blackjack(values, type_visit, axes[0], axes[1])

if not os.path.exists(save_dir):
	os.makedirs(save_dir)
plt.savefig(os.path.join(save_dir,"{}.png".format(type_visit)), dpi=fig.dpi)