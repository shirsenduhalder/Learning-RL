import gym
import time
import argparse
import numpy as np
import torch
import wrappers
import model

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
FPS = 25

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Enter Model File")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME, help="Enter environment name, default: {}".format(DEFAULT_ENV_NAME))
    parser.add_argument("-r", "--record", help="Enter directory to store recorded video")
    args = parser.parse_args()

    env = wrappers.make_env(args.env)
    if args.record:
        env = gym.wrappers.Monitor(env, args.record)
    
    net = model.DQN(env.observation_space.shape, env.action_space.n)
    net.load_state_dict(torch.load(args.model))

    state = env.reset()
    total_reward = 0.0

    while True:
        start_time = time.time()
        env.render()
        state_v = torch.tensor(np.array([state], copy=False))
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)

        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
        
        delta = 1/FPS - (time.time() - start_time)

        if delta > 0:
            time.sleep(delta)
    
    print("Total reward: {:2f}".format(total_reward))