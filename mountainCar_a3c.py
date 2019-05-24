import gym
import multiprocessing
import threading
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf

no_of_worker = multiprocessing.cpu_count()
no_ep_steps = 200
no_episodes = 2000
global_net_scope = 'Global_Net'

update_global = 10
gamma = 0.90
entropy_beta = 0.01
lr_a = 0.0001
lr_c = 0.001
render = True
log_dir = 'A3Clogs'

env = gym.make('MountainCarContinuous-v0')
env.reset()

no_of_states = env.observation_space.shape[0]
no_of_actions = env.action_space.shape[0]
action_bound = [env.action_space.low, env.action_space.high]

class ActorCritic(object):
	def __init__(self, scope, sess, globalAC=None):
		self.sess = sess
		self.actor_optimizer = tf.train.RMSPropOptimizer(lr_a, name='RMSPropA')
		self.critic_optimizer =tf.train.RMSPropOptimizer(lr_c, name='RMSPropC')

		if scope == global_net_scope:
			with tf.variable_scope(scope):
				self.s = tf.placeholder(tf.float32, [None, no_of_states], 'S')
				self.a_params, self.c_params = self._build_net(scope)[-2:]
		else:
			with tf.variable_scope(scope):
				self.s = tf.placeholder(tf.float32, [None, no_of_states], 'S')
				self.a_his = tf.placeholder(tf.float32, [None, no_of_actions], 'A')
				self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

				mean, var, self.v, self.a_params, self.c_params = self._build_net(scope)

				td = tf.subtract(self.v_target, self.v, name='TD_error')

				with tf.name_scope('critic_loss'):
					self.critic_loss = tf.reduce_mean(tf.square(td))

				with tf.name_scope('wrap_action'):
					mean, var = mean * action_bound[1], var + 1e-4

				normal_dist = tf.contrib.distributions.Normal(mean, var)

				with tf.name_scope('actor_loss'):
					log_prob = normal_dist.log_prob(self.a_his)
					exp_v = log_prob * td

					entropy = normal_dist.entropy()


					self.exp_v = exp_v + entropy_beta * entropy
					self.actor_loss = tf.reduce_mean(-self.exp_v)

				with tf.name_scope('choose_action'):
					self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), action_bound[0], action_bound[1])

				with tf.name_scope('local_grad'):
					self.a_grads = tf.gradients(self.actor_loss, self.a_params)
					self.c_grads = tf.gradients(self.critic_loss, self.c_params)

			with tf.name_scope('sync'):
				with tf.name_scope('pull'):
					self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
					self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]

				with tf.name_scope('push'):
					self.update_a_op = self.actor_optimizer.apply_gradients(zip(self.a_grads, globalAC.a_params))
					self.update_c_op = self.critic_optimizer.apply_gradients(zip(self.c_grads, globalAC.c_params))


	def _build_net(self, scope):

		w_init = tf.random_normal_initializer(0., 0.1)

		with tf.variable_scope('actor'):
			l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
			mean = tf.layers.dense(l_a, no_of_actions, tf.nn.tanh, kernel_initializer=w_init, name='mean')
			var = tf.layers.dense(l_a, no_of_actions, tf.nn.softplus, kernel_initializer=w_init,name='var')

		with tf.variable_scope('critic'):
			l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='l_c')
			v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')

		a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope+'/actor')
		c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope+'/critic')

		return mean, var, v, a_params, c_params

	def update_global(self, feed_dict):
		self.sess.run([self.update_a_op, self.update_c_op], feed_dict)

	def pull_global(self):
		self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

	def choose_action(self, s):
		s = s[np.newaxis, :]
		return self.sess.run(self.A, {self.s:s})[0]

class Worker(object):
	def __init__(self, name, globalAC, sess):
		self.env = gym.make('MountainCarContinuous-v0').unwrapped
		self.name = name

		self.AC = ActorCritic(name, sess, globalAC)
		self.sess = sess

	def work(self):
		global global_rewards, global_episodes
		total_step = 1

		buffer_s, buffer_a, buffer_r = [], [], []

		while not coord.should_stop() and global_episodes < no_episodes:

			s = self.env.reset()

			ep_r = 0
			for ep_t in range(no_ep_steps):
				if self.name == 'W_0' and render:
					self.env.render()

				a = self.AC.choose_action(s)

				s_, r, done, info = self.env.step(a)

				done = True if ep_t == no_ep_steps - 1 else False

				ep_r += r
				buffer_s.append(s)
				buffer_a.append(a)

				buffer_r.append((r + 8)/8)

				if total_step%update_global == 0 or done:
					if done:
						v_s_ = 0
					else:
						v_s_ = self.sess.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]

					buffer_v_target = []

					for r in buffer_r[::-1]:
						v_s = r + gamma * v_s_
						buffer_v_target.append(v_s)

					buffer_v_target.reverse()
					buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
					feed_dict = {
								self.AC.s: buffer_s,
								self.AC.a_his: buffer_a,
								self.AC.v_target: buffer_v_target
					}

					self.AC.update_global(feed_dict)
					buffer_s, buffer_a, buffer_r = [], [], []

					self.AC.pull_global()
				s = s_
				total_step += 1

				if done:
					if len(global_rewards) < 5:
						global_rewards.append(ep_r)
					else:
						global_rewards.append(ep_r)
						global_rewards[-1] = np.mean(global_rewards[-5:])

					global_episodes += 1
					break


global_rewards = []
global_episodes = 0
sess = tf.Session()

with tf.device("/cpu:0"):
	global_ac = ActorCritic(global_net_scope, sess)
	workers = []

	for i in range(no_of_worker):
		i_name = 'W_{}'.format(str(i))
		workers.append(Worker(i_name, global_ac, sess))


coord = tf.train.Coordinator()
sess.run(tf.global_variables_initializer())

if os.path.exists(log_dir):
	shutil.rmtree(log_dir)

tf.summary.FileWriter(log_dir, sess.graph)
worker_threads = []

for worker in workers:
	job = lambda: worker.work()
	t = threading.Thread(target=job)
	t.start()
	worker_threads.append(t)
coord.join(worker_threads)