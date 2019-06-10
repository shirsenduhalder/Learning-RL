import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake8x8-v0"
GAMMA = 0.90
TEST_EPISODES = 400

class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transitions = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)
    
    def play_n_random_steps(self, count):
        # used for gathering data about transitions and rewards
        for _ in range(count):
            action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action)
            self.rewards[(self.state, action, next_state)] = reward
            self.transitions[(self.state, action)][next_state] += 1
            self.state = self.env.reset() if done else next_state
    
    def calc_action_value(self, state, action):
        # for calculation of the q value wrt a state and action
        target_counts = self.transitions[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0

        for target_state, count in target_counts.items():
            reward = self.rewards[(state, action, target_state)]
            action_value += (count/total) * (reward + GAMMA*self.values[target_state])
        
        return action_value
    
    def select_action(self, state):
        # selection of the action with the highest q value
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)

            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        
        return best_action
    
    def play_episode(self, env):
        # we use a different env here as this function is used to play test episodes during which we do not want to mess up with the current state of the main environment used for gathering data
        total_reward = 0.0
        state = env.reset()

        while True:
            action = self.select_action(state)
            next_state, reward, done, _ = env.step(action)
            self.rewards[(state, action, next_state)] = reward
            self.transitions[(state, action)][next_state] += 1

            total_reward += reward

            if done:
                break
            
            state = next_state
        
        return total_reward
    
    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            state_values = [self.calc_action_value(state, action) for action in range(self.env.action_space.n)]
            self.values[state] = max(state_values)


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(logdir="logs", comment="-v-iteration")

    iter_no = 0
    best_reward = 0.0

    while True:
        iter_no += 1
        agent.play_n_random_steps(200)
        agent.value_iteration()

        reward = 0.0

        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)

        if reward > best_reward:
            print("Best reward updated: {} --> {}".format(best_reward, reward))

            best_reward = reward
        
        if reward > 0.80:
            print("Solved in {} iterations".format(iter_no))

            break
    
    writer.close()