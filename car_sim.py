import gym
env = gym.make('CarRacing-v0')
env.reset()
for _ in range(1000):
    env.render()
    obs, rew, done, info = env.step(env.action_space.sample())

    if done:
    	env.reset()

env.close()