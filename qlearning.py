import numpy as np
class QLearning:
    def __init__(self, actions, learning_rate=0.5, discount_factor=0.95, exploration_rate=0.1):
        self.actions = actions  # list of actions
        self.lr = learning_rate  # learning rate
        self.df = discount_factor  # discount factor
        self.er = exploration_rate  # exploration rate
        self.q_table = {}  # Q-table

    def choose_action(self, state):
        if np.random.uniform() < self.er:
            # Explore: select a random action
            action = np.random.choice(self.actions)
        else:
            # Exploit: select the action with max value (greedy policy)
            state_action = self.q_table[state]
            action = self.arg_max(state_action)
        return action

    def learn(self, state, action, reward, next_state):
        q_predict = self.q_table[state, action]
        if next_state != 'terminal':
            q_target = reward + self.df * np.max(self.q_table[next_state])
        else:
            q_target = reward  # next state is terminal
        # Update Q-table
        self.q_table[state, action] += self.lr * (q_target - q_predict)

    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return np.random.choice(max_index_list)
