import gym
import numpy as np
import sys

env = gym.make("FrozenLake-v0")

def compute_value_function(env, policy, gamma=1.0):
    env = env.unwrapped
    value_table = np.zeros(env.nS)
    threshold = 1e-10
    while True:
        updated_value_table = np.copy(value_table)
        for state in range(env.nS):
            action = policy[state]
            value_table[state] = sum([trans_prob * (reward_prob + gamma * updated_value_table[next_state]) for trans_prob, next_state, reward_prob, _ in env.P[state][action]])
            
        if (np.sum((np.fabs(updated_value_table - value_table))) <= threshold):
            break
    return value_table

def policy_extraction(env, value_table, gamma=1.0):
    env = env.unwrapped
    policy = np.zeros(env.nS)

    for state in range(env.nS):
        Q_table = np.zeros(env.nA)
        for action in range(env.nA):
            for next_sr in env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                Q_table[action] += trans_prob * (reward_prob + gamma*value_table[next_state])
        policy[state] = np.argmax(Q_table)

    return policy

def policy_iteration(env, gamma=1.0):
    old_policy = np.zeros(env.observation_space.n)
    no_of_iterations = 200000
    gamma = 1.0

    for i in range(no_of_iterations):
        sys.stdout.write("\rIteration {}/{}".format(i+1, no_of_iterations))
        sys.stdout.flush()
        new_value_function = compute_value_function(env, old_policy, gamma)
        new_policy = policy_extraction(env, new_value_function, gamma)
        
        if (np.all(old_policy == new_policy)):
            print('\nPolicy Iteration converged. ITERATION: {}'.format(i + 1))
            break

        old_policy = new_policy

    return new_policy

print(policy_iteration(env))