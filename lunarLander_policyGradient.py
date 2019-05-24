import tensorflow as tf
import numpy as np
import gym
from tensorflow.python.framework import ops
import time

class PolicyGradient:
	def __init__(self, n_x, n_y, learning_rate=0.01, reward_decay=0.95):

		self.n_x = n_x
		self.n_y = n_y
		self.lr = learning_rate

		self.gamma = reward_decay

		self.episode_obs, self.episode_rewards, self.episode_actions = [], [], []

		self.build_network()

		self.cost_history = []
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def store_transition(self, s, a, r):
		self.episode_obs.append(s)
		self.episode_rewards.append(r)

		action = np.zeros(self.n_y)
		action[a] = 1
		self.episode_actions.append(action)

	def choose_action(self, observation):
		observation = observation[:, np.newaxis]

		prob_weights = self.sess.run(self.outputs_softmax, feed_dict={self.X:observation})

		action = np.random.choice(range(len(prob_weights.ravel())), p=prob_weights.ravel())

		return action

	def build_network(self):

		self.X = tf.placeholder(tf.float32, shape=(self.n_x, None), name='X')
		self.Y = tf.placeholder(tf.float32, shape=(self.n_y, None), name='Y')
		self.discounted_episode_rewards = tf.placeholder(tf.float32, [None, ], name="action_value")

		n_h1 = 10
		n_h2 = 10
		n_output = self.n_y

		initializer = tf.contrib.layers.xavier_initializer(seed=1)

		W1 = tf.Variable(initializer(shape=(n_h1, self.n_x)), name="W1", dtype=tf.float32)
		b1 = tf.Variable(initializer(shape=(n_h1, 1)), name="b1", dtype=tf.float32)

		W2 = tf.Variable(initializer(shape=(n_h2, n_h1)), name="W2", dtype=tf.float32)
		b2 = tf.Variable(initializer(shape=(n_h2, 1)), name="b2", dtype=tf.float32)

		W3 = tf.Variable(initializer(shape=(n_output, n_h2)), name="W3", dtype=tf.float32)
		b3 = tf.Variable(initializer(shape=(n_output, 1)), name="b3", dtype=tf.float32)

		Z1 = tf.add(tf.matmul(W1, self.X), b1)
		A1 = tf.nn.relu(Z1)

		Z2 = tf.add(tf.matmul(W2, A1), b2)
		A2 = tf.nn.relu(Z2)

		Z3 = tf.add(tf.matmul(W3, A2), b3)
		# A3 = tf.nn.softmax(Z3)

		logits = tf.transpose(Z3)
		labels = tf.transpose(self.Y)
		self.outputs_softmax = tf.nn.softmax(logits, name='A3')

		neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)

		loss = tf.reduce_mean(neg_log_prob * self.discounted_episode_rewards)

		self.train_op= tf.train.AdamOptimizer(self.lr).minimize(loss)

	def discount_and_norm_rewards(self):
		discounted_episode_rewards = np.zeros_like(self.episode_rewards)
		cumulative = 0

		for t in reversed(range(len(self.episode_rewards))):
			cumulative = cumulative * self.gamma + self.episode_rewards[t]
			discounted_episode_rewards[t] = cumulative

		discounted_episode_rewards -= np.mean(discounted_episode_rewards)
		discounted_episode_rewards /= np.std(discounted_episode_rewards)

		return discounted_episode_rewards

	def learn(self):
		discounted_episode_rewards = self.discount_and_norm_rewards()

		self.sess.run(self.train_op, feed_dict={self.X:np.vstack(self.episode_obs).T, self.Y:np.vstack(self.episode_actions).T, self.discounted_episode_rewards:discounted_episode_rewards})

		self.episode_obs, self.episode_rewards, self.episode_actions = [], [], []

		return discounted_episode_rewards


env = gym.make('LunarLander-v2')
env = env.unwrapped

RENDER = True
NUM_EPISODES = 5000
rewards = []
RENDER_REWARD_MIN = 5000

PG = PolicyGradient(n_x=env.observation_space.shape[0], n_y=env.action_space.n, learning_rate=0.02, reward_decay=0.99)

for episode in range(NUM_EPISODES):
	observation = env.reset()
	episode_reward = 0

	while True:
		if RENDER:
			env.render()

		action = PG.choose_action(observation)

		observation_, reward, done, info = env.step(action)
		PG.store_transition(observation, action, reward)

		episode_rewards_sum = sum(PG.episode_rewards)

		if episode_rewards_sum < -250:
			done=True

		if done:
			episode_rewards_sum = sum(PG.episode_rewards)
			rewards.append(episode_rewards_sum)
			max_reward = np.amax(rewards)

			print("Episode: {}, Reward:{}, Max Reward so far: {}".format(episode, episode_rewards_sum, max_reward))

			discounted_episode_rewards = PG.learn()

			if max_reward > RENDER_REWARD_MIN:
				RENDER = False

			break

		observation = observation_