import numpy as np

BOARD_ROWS = 3
BOARD_COLS = 4
WIN_STATE = (0, 3)
LOSE_STATE = (1, 3)
START = (2, 0)
DETERMINISTIC = False

def compute_position(s):
    return np.array([s // 4, s % 4])

def compute_distance(i, j):
    if np.sum(np.abs(compute_position(i) - compute_position(j))) <= 1:
        return 1
    else:
        return 0

def reverse_position(array):
    return array[0] * 4 + array[1]

def policy_to_action(policy):
    action_list = ["↑", "←", "→", "↓"]
    action = np.argmax(policy, 1)
    return [action_list[i] for i in action]

def run_grid_world(policy, state_value, action_value, rounds=100):
    print("Let's run grid world!")
    agent = Agent(policy)
    agent.play(rounds)
    agent.show_policy()
    print("\nstate value:\n {}".format(np.round(state_value, 3)))
    print("action value:\n {}".format(np.round(action_value, 3)))

class environment:
    def __init__(self, state=START):
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        self.board[1, 1] = -1
        self.state = state
        self.isEnd = False
        self.determine = DETERMINISTIC

    def giveReward(self):
        if self.state == WIN_STATE:
            return 1
        elif self.state == LOSE_STATE:
            return -1
        else:
            return 0

    def isEndFunc(self):
        if (self.state == WIN_STATE) or (self.state == LOSE_STATE):
            self.isEnd = True

    def _chooseActionProb(self, action):
        if action == "up":
            return np.random.choice(["up", "left", "right"], p=[0.8, 0.1, 0.1])
        if action == "down":
            return np.random.choice(["down", "left", "right"], p=[1.0, 0.0, 0.0])
        if action == "left":
            return np.random.choice(["left", "up", "down"], p=[1.0, 0.0, 0.0])
        if action == "right":
            return np.random.choice(["right", "up", "down"], p=[1.0, 0.0, 0.0])

    def nextPosition(self, action):
        """
        action: up, down, left, right
        -------------
        0 | 1 | 2| 3|
        1 |
        2 |
        return next position on board
        """
        if self.determine:
            if action == "up":
                nextState = (self.state[0] - 1, self.state[1])
            elif action == "down":
                nextState = (self.state[0] + 1, self.state[1])
            elif action == "left":
                nextState = (self.state[0], self.state[1] - 1)
            else:
                nextState = (self.state[0], self.state[1] + 1)
            self.determine = False
        else:
            # non-deterministic
            action = self._chooseActionProb(action)
            self.determine = True
            nextState = self.nextPosition(action)

        # if next state is legal
        if (nextState[0] >= 0) and (nextState[0] <= 2):
            if (nextState[1] >= 0) and (nextState[1] <= 3):
                if nextState != (1, 1):
                    return nextState
        return self.state

    def showBoard(self):
        self.board[self.state] = 1
        for i in range(0, BOARD_ROWS):
            print('-----------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = '*'
                if self.board[i, j] == -1:
                    token = 'z'
                if self.board[i, j] == 0:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-----------------')
    
    def show_action_board(self, action):
        
        for i in range(0, BOARD_ROWS):
            print('-----------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if 4*i + j == 3:
                    token = 'G'
                if 4*i + j == 7:
                    token = 'T'
                if self.board[i, j] == -1:
                    token = 'z'
                else:
                    token = action[4*i + j]
                out += token + ' | '
            print(out)
        print('-----------------')

class randomAgent:
    def __init__(self):
        self.states = []  # record position and action taken at the position
        self.actions = ["up", "down", "left", "right"]
        self.env = environment()
        self.isEnd = self.env.isEnd
        self.result_stat = []

    def randomAction(self):
        action = np.random.choice(self.actions)
        return action

    def takeAction(self, action):
        position = self.env.nextPosition(action)
        return environment(state=position)

    def reset(self):
        self.states = []
        self.env = environment()
        self.isEnd = self.env.isEnd

    def play(self, rounds):
        i=0
        self.result_stat = []
        while i < rounds:
            if self.env.isEnd:
                reward = self.env.giveReward()
                if reward >= 1:
                    self.result_stat.append(reward)
                self.reset()
                i += 1
            else:
                action = self.randomAction()
                self.states.append([(self.env.state), action])
                self.env = self.takeAction(action)
                self.env.isEndFunc()
                self.isEnd = self.env.isEnd
        success_rate = np.sum(self.result_stat) / rounds
        print("Success rate:{} %".format(success_rate * 100))

class Agent:
    def __init__(self, policy, epsilon=None):
        self.states = []  # record position and action taken at the position
        self.actions = ["up", "left", "right", "down"]
        self.env = environment()
        self.isEnd = self.env.isEnd
        self.result_stat = []
        self.policy = policy
        self.history = []
        self.epsilon=epsilon # epsilon-soft

    def Action(self, state):
        state = reverse_position(state)
        action = np.random.choice(self.actions, p=self.policy[state,:])
        if self.epsilon is not None:
            prob = np.random.uniform(0, 1)
            if prob < self.epsilon: # epsilon-soft policy
                action = np.random.choice(self.actions)
        return action

    def takeAction(self, action):
        position = self.env.nextPosition(action)
        return environment(state=position)

    def reset(self):
        self.states = []
        self.env = environment()
        self.isEnd = self.env.isEnd

    def play(self, rounds, stat=False):
        i=0
        self.result_stat = []
        while i < rounds:
            if self.env.isEnd:
                reward = self.env.giveReward()
                self.history.append(self.states)
                self.result_stat.append(reward)
                self.reset()
                i += 1
            else:
                action = self.Action(self.env.state)
                self.states.append([(self.env.state), action])
                self.env = self.takeAction(action)
                self.env.isEndFunc()
                self.isEnd = self.env.isEnd
        
        success_rate = self.result_stat.count(1) / rounds
        if stat: # stat mode
            return success_rate
        else:
            return self.history, self.result_stat, success_rate

    def show_policy(self):
        action = policy_to_action(self.policy)
        self.env.show_action_board(action)