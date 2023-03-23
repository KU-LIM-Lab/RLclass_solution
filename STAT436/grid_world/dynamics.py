import numpy as np

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

class dynamics:
    def __init__(self):
        # state, next_state, action, reward
        self.dynamics = np.zeros((12, 12, 4, 3)) 
        self.dynamics = self.return_dynamics()
        
    def return_dynamics(self):
        for i in range(12):
            # press 'up'
            if compute_position(i)[0] == 0: # top row
                self.dynamics[i, i, 0, 1] += 0.8
            elif i == 9: # wall
                self.dynamics[i, i, 0, 1] += 0.8
            else:
                if i not in [3, 5, 7]:
                    position = compute_position(i) - np.array([1, 0])
                    self.dynamics[i, reverse_position(position), 0, 1] += 0.8
            # press 'left'
            if compute_position(i)[1] == 0: # top left
                self.dynamics[i, i, 0, 1] += 0.1
                self.dynamics[i, i, 1, 1] += 1
            elif i == 6: # wall
                self.dynamics[i, i, 0, 1] += 0.1
                self.dynamics[i, i, 1, 1] += 1
            else:
                if i not in [3, 5, 7]:
                    position = compute_position(i) - np.array([0, 1])
                    self.dynamics[i, reverse_position(position), 0, 1] += 0.1
                    self.dynamics[i, reverse_position(position), 1, 1] += 1
            # press 'right'
            if compute_position(i)[1] == 3: # top-right column
                self.dynamics[i, i, 0, 1] += 0.1
                self.dynamics[i, i, 2, 1] += 1
            elif i == 4: # wall
                self.dynamics[i, i, 0, 1] += 0.1
                self.dynamics[i, i, 2, 1] += 1
            else:
                position = compute_position(i) + np.array([0, 1])
                if i not in [3, 5, 7]:
                    self.dynamics[i, reverse_position(position), 0, 1] += 0.1
                    self.dynamics[i, reverse_position(position), 2, 1] += 1
            # press 'down'
            if compute_position(i)[0] == 2: # buttom row
                self.dynamics[i, i, 3, 1] += 1
            elif i == 1: # wall
                self.dynamics[i, i, 3 ,1] += 1
            else:
                if i not in [3, 5, 7]:
                    position = compute_position(i) + np.array([1, 0])
                    self.dynamics[i, reverse_position(position), 3, 1] += 1
            if i in [3, 5, 7]: # end, wall cases
                self.dynamics[i, :, :, :] = 0 # absorbing state
                if i == 3:
                    self.dynamics[i, 8, :, 0] = 1 # goal state
                if i == 5:
                    self.dynamics[i, i, :, 1] = 1 # wall
                if i == 7:
                    self.dynamics[i, 8, :, 2] = 1 # trap state
        return self.dynamics