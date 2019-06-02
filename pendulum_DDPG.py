import tensorflow as tf
import numpy as np
import gym

episode_steps = 500
lr_a = 0.001
lr_c = 0.002
gamma = 0.90
alpha = 0.01
memory_size = 10000
batch_size = 32

class DDPG(object):
	