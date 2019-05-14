import sys, os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from vizdoom import *
import argparse
import random
import time, timeit
import math
import numpy as np

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', "--show-demo", help="Boolean to show demo", default=False)

	return parser.parse_args()

def demo(show_demo):
	game = DoomGame()
	game.load_config("basic.cfg")
	game.init()

	shoot = [0, 0, 1]
	left = [1, 0, 0]
	right = [0, 1, 0]
	actions = [shoot, left, right]

	no_of_episodes = 10

	for i in range(no_of_episodes):
		game.new_episode()
		episodic_reward = 0
		while not game.is_episode_finished():
			state = game.get_state()
			img = state.screen_buffer

			misc = state.game_variables
			reward = game.make_action(random.choice(actions))

			episodic_reward+=reward
		print("Episode Number: {}, Reward: {}".format(i + 1, episodic_reward))
		time.sleep(2)

def get_input_shape(Image, Filter, Stride):
	layer1 = math.ceil(((Image - Filter + 1)/Stride))
	o1 = math.ceil((layer1/Stride))

	layer2 = math.ceil(((o1 - Filter + 1)/Stride))
	o2 = math.ceil((layer2/Stride))

	layer3 = math.ceil(((o2 - Filter + 1)/Stride))
	o3 = math.ceil((layer3/Stride))

	return int(o3)

class DRQN():
	def __init__(self, input_shape, num_actions, initial_learning_rate):
		self.tfcast_type = tf.float32
		self.input_shape = input_shape
		self.num_actions = num_actions
		self.learning_rate = initial_learning_rate

		#Convolution Hyperparams
		self.filter_size = 5
		self.num_filters = [16, 32, 64]
		self.stride = 2
		self.poolsize = 2
		self.convolution_shape = get_input_shape(input_shape[0], self.filter_size, self.stride) * get_input_shape(input_shape[1], self.filter_size, self.stride) * self.num_filters[2]

		#RNN hyperparameters
		self.cell_size = 100 #no of neurons
		self.hidden_layer = 50 #no of hidden layers
		self.dropout_probability = [0.3, 0.2]

		#Optimization hyperparameters
		self.loss_decay_rate = 0.96
		self.loss_decay_steps = 180

		self.input = tf.placeholder(shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype=self.tfcast_type)
		self.target_vector = tf.placeholder(shape=(self.num_actions, 1), dtype=self.tfcast_type)
		initial_shape = (1, self.input_shape[0], self.input_shape[1], self.input_shape[2])

		#Feature maps initialization for CNN
		self.features1 = tf.Variable(tf.random_normal([self.filter_size, self.filter_size, self.input_shape[2], self.num_filters[0]]), dtype=self.tfcast_type)
		self.features2 = tf.Variable(tf.random_normal([self.filter_size, self.filter_size, self.num_filters[0], self.num_filters[1]]), dtype=self.tfcast_type)
		self.features3 = tf.Variable(tf.random_normal([self.filter_size, self.filter_size, self.num_filters[1], self.num_filters[2]]), dtype=self.tfcast_type)

		###building feed forward CNN
		#Layer 1
		self.conv1 = tf.nn.conv2d(input=tf.reshape(self.input, shape=initial_shape), filter=self.features1, strides=[1, self.stride, self.stride, 1], padding='VALID')
		self.relu1 = tf.nn.relu(self.conv1)
		self.pool1 = tf.nn.max_pool(self.relu1, ksize=[1, self.poolsize, self.poolsize, 1], strides=[1, self.stride, self.stride, 1], padding='SAME')

		#Layer 2
		self.conv2 = tf.nn.conv2d(input=self.pool1, filter=self.features2, strides=[1, self.stride, self.stride, 1], padding='VALID')
		self.relu2 = tf.nn.relu(self.conv2)
		self.pool2 = tf.nn.max_pool(self.relu2, ksize=[1, self.poolsize, self.poolsize, 1], strides=[1, self.stride, self.stride, 1], padding='SAME')

		#Layer 3
		self.conv3 = tf.nn.conv2d(input=self.pool2, filter=self.features3, strides=[1, self.stride, self.stride, 1], padding='VALID')
		self.relu3 = tf.nn.relu(self.conv3)
		self.pool3 = tf.nn.max_pool(self.relu3, ksize=[1, self.poolsize, self.poolsize, 1], strides=[1, self.stride, self.stride, 1], padding='SAME')

		#dropout
		self.drop1 = tf.nn.dropout(self.pool3, self.dropout_probability[0])
		self.reshaped_input = tf.reshape(self.drop1, shape=[1, -1])


		#initialize variables for RNN
		self.h = tf.Variable(initial_value=np.zeros((1, self.cell_size)), dtype=self.tfcast_type)

		#hidden to hidden weight matrix
		self.rW = tf.Variable(initial_value=np.random.uniform(low = -np.sqrt(6. / (self.convolution_shape + self.cell_size)), high = np.sqrt(6. / (self.convolution_shape + self.cell_size)), size = (self.convolution_shape, self.cell_size)), dtype = self.tfcast_type)

		#input to hidden
		self.rU = tf.Variable(initial_value=np.random.uniform(low = -np.sqrt(6. / (2 * self.cell_size)), high = np.sqrt(6. / (2 * self.cell_size)), size = (self.cell_size, self.cell_size)), dtype = self.tfcast_type)

		#hidden to output
		self.rV = tf.Variable(initial_value = np.random.uniform(low = -np.sqrt(6. / (2 * self.cell_size)), high = np.sqrt(6. / (2 * self.cell_size)), size = (self.cell_size, self.cell_size)), dtype = self.tfcast_type)

		#bias
		self.rb = tf.Variable(initial_value=np.zeros(self.cell_size), dtype=self.tfcast_type)
		self.rc = tf.Variable(initial_value=np.zeros(self.cell_size), dtype=self.tfcast_type)

		self.step_count = tf.Variable(initial_value=0, dtype=self.tfcast_type)
		self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.step_count, self.loss_decay_steps, self.loss_decay_rate, staircase=False)

		#Weights of feed forward network
		self.fW = tf.Variable(initial_value = np.random.uniform(low = -np.sqrt(6. / (self.cell_size + self.num_actions)), high = np.sqrt(6. / (self.cell_size + self.num_actions)), size = (self.cell_size, self.num_actions)), dtype = self.tfcast_type)
		self.fb = tf.Variable(initial_value=np.zeros(self.num_actions), dtype=self.tfcast_type)

		self.h = tf.tanh(tf.matmul(self.reshaped_input, self.rW) + tf.matmul(self.h, self.rU) + self.rb)
		self.o = tf.nn.softmax(tf.matmul(self.h, self.rV) + self.rc)

		self.drop2 = tf.nn.dropout(self.o, self.dropout_probability[1])
		self.output = tf.reshape(tf.matmul(self.drop2, self.fW) + self.fb, shape=[-1, 1])
		self.prediction = tf.argmax(self.output)

		self.loss = tf.reduce_mean(tf.square(self.target_vector - self.output))
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
		self.gradients = self.optimizer.compute_gradients(self.loss)
		self.update = self.optimizer.apply_gradients(self.gradients)

		self.parameters = (self.features1, self.features2, self.features3, self.rW, self.rU, self.rV, self.rb, self.fW, self.fb)

class ExperienceReplay():
	def __init__(self, buffer_size):
		self.buffer = []
		self.buffer_size = buffer_size

	def appendToBuffer(self, memory_tuplet):
		if len(self.buffer) > self.buffer_size:
			for i in range(len(self.buffer) - self.buffer_size):
				self.buffer.remove(self.buffer[0])

		self.buffer.append(memory_tuplet)

	def sample(self, n):
		memories = []
		for i in range(n):
			memory_index = np.random.randint(0, len(self.buffer))
			memories.append(self.buffer[memory_index])

		return memories

def train(num_episodes, episode_length, learning_rate, scenario="deathmatch.cfg", map_path="map02", render=False):

	discount_factor = 0.99
	sample_frequency = 5
	store_frequency = 50

	total_reward = 0
	total_loss = 0
	old_q_value = 0

	rewards = []
	losses = []

	game = DoomGame()
	game.set_doom_scenario_path(scenario)
	game.set_doom_map(map_path)

	game.set_screen_resolution(ScreenResolution.RES_256X160)
	game.set_screen_format(ScreenFormat.RGB24)

	game.set_render_hud(False)
	game.set_render_minimal_hud(False)
	game.set_render_crosshair(False)
	game.set_render_weapon(True)
	game.set_render_particles(False)
	game.set_render_decals(False)
	game.set_render_effects_sprites(False)
	game.set_render_messages(False)
	game.set_render_corpses(False)
	game.set_render_screen_flashes(False)

	game.add_available_button(Button.MOVE_LEFT)
	game.add_available_button(Button.MOVE_RIGHT)
	game.add_available_button(Button.TURN_LEFT)
	game.add_available_button(Button.TURN_RIGHT)
	game.add_available_button(Button.MOVE_FORWARD)
	game.add_available_button(Button.MOVE_BACKWARD)
	game.add_available_button(Button.ATTACK)

	game.add_available_button(Button.TURN_LEFT_RIGHT_DELTA, 90)
	game.add_available_button(Button.LOOK_UP_DOWN_DELTA, 90)

	actions = np.zeros((game.get_available_buttons_size(), game.get_available_buttons_size()))
	count = 0
	for i in actions:
		i[count] = 1
		count += 1
	actions = actions.astype(int).tolist()

	game.add_available_game_variable(GameVariable.AMMO0)
	game.add_available_game_variable(GameVariable.HEALTH)
	game.add_available_game_variable(GameVariable.KILLCOUNT)

	game.set_episode_timeout(6 * episode_length)
	game.set_episode_start_time(10)
	game.set_window_visible(render)

	game.set_sound_enabled(True)

	game.set_living_reward(0)

	game.set_mode(Mode.PLAYER)
	game.init()

	actionDRQN = DRQN((160, 256, 3), game.get_available_buttons_size() - 2, learning_rate)
	targetDRQN = DRQN((160, 256, 3), game.get_available_buttons_size() - 2, learning_rate)

	experiences = ExperienceReplay(1000)

	saver = tf.train.Saver({v.name: v for v in actionDRQN.parameters}, max_to_keep=1)

	loss_summary = tf.summary.scalar('Loss', actionDRQN.loss)
	summary_merge = tf.summary.merge_all()
	file_writer = tf.summary.FileWriter('DRQN_logs', tf.get_default_graph())

	with tf.Session() as sess:

		sess.run(tf.global_variables_initializer())
		summary_frame_count = 1
		for episode in range(num_episodes):
			game.new_episode()

			for frame in range(episode_length):
				state = game.get_state()
				s = state.screen_buffer

				a = actionDRQN.prediction.eval(feed_dict={actionDRQN.input:s})[0]
				action = actions[a]

				reward = game.make_action(action)

				total_reward += reward
				if game.is_episode_finished():
					break
				if (frame%store_frequency) == 0:
					experiences.appendToBuffer((s, action, reward))

				if (frame%sample_frequency) == 0:
					memory = experiences.sample(1)
					mem_frame = memory[0][0]
					mem_reward = memory[0][2]

					Q1 = actionDRQN.output.eval(feed_dict={actionDRQN.input:mem_frame})
					Q2 = targetDRQN.output.eval(feed_dict={targetDRQN.input:mem_frame})

					learning_rate = actionDRQN.learning_rate.eval()

					Qtarget = old_q_value + learning_rate * (mem_reward + discount_factor*Q2 - old_q_value)
					old_q_value = Qtarget

					loss = actionDRQN.loss.eval(feed_dict={actionDRQN.target_vector:Qtarget, actionDRQN.input:mem_frame})
					total_loss += loss
					
					summary_write = summary_merge.eval(feed_dict={actionDRQN.target_vector:Qtarget, actionDRQN.input:mem_frame})
					file_writer.add_summary(summary_write, summary_frame_count)
					
					actionDRQN.update.run(feed_dict={actionDRQN.target_vector:Qtarget, actionDRQN.input:mem_frame})
					targetDRQN.update.run(feed_dict={targetDRQN.target_vector:Qtarget, targetDRQN.input:mem_frame})
					summary_frame_count+=1

			
			rewards.append((episode, total_reward))
			losses.append((episode, total_loss))


			print("Episode: {}, Reward: {}, Loss: {}".format(episode + 1, total_reward, total_loss))

			total_loss, total_reward = 0, 0

train(num_episodes=10000, episode_length=300, learning_rate=0.001, render=True)
args = parse_arguments()