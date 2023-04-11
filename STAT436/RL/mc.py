import numpy as np

def compute_position(state):
    # state(12) -> array(3, 4)
    return np.array([state // 4, state % 4])

def reverse_position(array):
    # array(3, 4) -> state(12)
    return array[0] * 4 + array[1]

def create_value_memory(dim=(12, 4)):
    value_memory = np.zeros(dim)
    return value_memory

def action_to_index(action):
    # up, left, right, down
    if action == 'up':
        return 0
    if action == 'left':
        return 1
    if action == 'right':
        return 2
    if action == 'down':
        return 3
    else:
        raise ValueError('not proper action')

def transform_trajectory_memory(trajectory):
    # trajectory -> {(state, action)}
    trajectory_ = [(reverse_position(pair[0]), action_to_index(pair[1])) for pair in trajectory]

    return trajectory_

def mc_eval(history, reward_stat, gamma, update=0.99):
    """
    input: history
    output: value estimation
    """
    value_memory = create_value_memory()
    
    for trajectory, reward in zip(history, reward_stat):
        
        trajectory_ = transform_trajectory_memory(trajectory)
        T = len(trajectory_) - 1
        G = reward

        tmp = np.zeros_like(value_memory)
        tmp[trajectory_[T][0], trajectory_[T][1]] = G

        for t in range(1, T+1, 1):
            G = gamma * G
            tmp[trajectory_[T-t][0], trajectory_[T-t][1]] = G
        
        value_memory = value_memory + update * (tmp - value_memory)

    return value_memory

# from Policy Iteration
def one_hot(scalar, dim):
    vec = np.zeros(dim)
    vec[scalar] = 1
    return vec

def greedy_action(array, dim):
    vec = np.zeros(dim)
    array_size = array.shape[0]
    for _ in array:
        vec[_] = 1 / array_size

    return vec

def argmax(vec, tie=True):
    if tie:
        return np.where(vec == np.max(vec))[0]
    else: # ordinary argmax
        return np.argmax(vec)

# update policy w/ greedy policy
def update_policy(policy, action_value):

    greedy_policy = np.zeros_like(policy)

    for state in range(12):

        action = argmax(action_value[state, :])
        action = greedy_action(action, 4)
        greedy_policy[state] = action

    return greedy_policy

# Policy Improvement w/ MC
def mc_policy_iteration(pi_init, agent, gamma, eps=1e-8, play_num=100, epsilon=None):

    # call policy eval
    pi = pi_init
    agent_ = agent(pi_init)
    history, reward_stat, _ = agent_.play(play_num, stat=False)
    action_value = mc_eval(history, reward_stat, gamma)

    advances = np.inf
    n_it = 0

    while advances > eps or n_it <= 2:
        
        # policy improvement
        pi_new = update_policy(pi, action_value)

        # policy evaluation
        agent_ = agent(pi_new, epsilon)
        history, reward_stat, success_rate = agent_.play(play_num, stat=True)
        action_value_new = mc_eval(history, reward_stat, gamma)

        # stop condition
        advances = action_value_new - action_value
        # advances = advances * (advances > 0)
        advances = np.abs(action_value_new - action_value)
        advances = np.sum(advances)

        # save policy and update values
        pi = pi_new
        action_value = action_value_new
        n_it += 1
        epsilon /= n_it

        if n_it % 10 == 0:
            print("Iteration: {}, Success rate:{} %, Error: {}, eps: {}".format(play_num * n_it, success_rate * 100, advances, epsilon))

    print("Monte-Carlo Policy Iteration converged. (Iteration={}, Error={})".format(play_num * n_it, advances))

    return pi_new, action_value_new