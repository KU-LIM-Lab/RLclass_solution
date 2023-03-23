import numpy as np

class pi_dynamics:
    def __init__(self, pi, gamma, reward, dynamics):
        self.pi = pi    
        self.gamma = gamma
        self.reward = reward
        grid_world_dynamics = dynamics()
        self.dynamics = grid_world_dynamics.dynamics
        # w/o tensor notation
        self.pi_dynamics = np.zeros_like(self.dynamics) # [current_state, next_state, action, value] 
        self.P_reward = np.zeros((12, 3))
        self.P_value = np.zeros((12, 12))
        self.pi_dynamics, self.P_reward, self.P_value = self.return_all()

    def update_pi_dynamics(self, pi_dynamics):
        for i in range(12):
            for a in range(4):
                # broadcasting
                pi_dynamics[i, :, a, :] = self.pi[i, a] * self.dynamics[i, :, a, :]
        return pi_dynamics

    def compute_P_reward(self, P_reward):
        # state -> reward
        for j in range(12):
             for r in range(3):
                 # marginalization
                 P_reward[j, r] = np.sum(self.pi_dynamics[j, :, :, r])
        return P_reward

    def compute_P_value(self, P_value):
        # state -> state
        for j in range(12):
             for i in range(12):
                 # marginalization
                 P_value[j, i] = np.sum(self.pi_dynamics[j, i, :, :])
        return P_value

    def return_all(self):
        return self.update_pi_dynamics(self.pi_dynamics), self.compute_P_reward(self.P_reward), self.compute_P_value(self.P_value)

    def compute_state_value(self):
        coeff = np.eye(12) - self.gamma * self.P_value
        inv_coeff = np.linalg.inv(coeff)
        state_value = inv_coeff @ self.P_reward @ self.reward
        return state_value

    def compute_action_value(self):
        state_value = self.compute_state_value()
        expectation_reward = np.zeros((12, 4))
        expectation_value = np.zeros((12, 4))
        for i in range(12):
            for a in range(4):
                expectation_reward[i, a] = self.reward @ np.sum(self.dynamics, axis=1)[i, a, :]
                expectation_value[i, a] = self.gamma * state_value @ np.sum(self.dynamics, axis=3)[i, :, a]
        action_value = expectation_reward + expectation_value
        return action_value

def one_hot(scalar, dim):
    vec = np.zeros(dim)
    vec[scalar] = 1
    return vec

# update policy w/ greedy policy
def update_policy(policy, action_value):

    greedy_policy = np.zeros_like(policy)

    for state in range(12):
        if state in [3, 5, 7]:
            action = np.array([0.25, 0.25, 0.25, 0.25])
        else:
            action = np.argmax(action_value[state, :])
            action = one_hot(action, 4)
        greedy_policy[state] = action
    
    return greedy_policy

# policy iteration
def policy_iteration(pi, gamma, reward, dynamics, eps=1e-8):

    init_dynamics = dynamics
    init_pi_dynamics = pi_dynamics(pi, gamma, reward, init_dynamics)
    action_value = init_pi_dynamics.compute_action_value()

    pi_old = pi
    pi_new = update_policy(pi, action_value)

    advances = np.inf
    n_it = 0

    while np.sum(advances) > eps:
        dynamics_old = pi_dynamics(pi=pi_old, gamma=gamma, reward=reward, dynamics=dynamics)
        dynamics_new = pi_dynamics(pi=pi_new, gamma=gamma, reward=reward, dynamics=dynamics)
        state_value_old = dynamics_old.compute_state_value()
        state_value_new = dynamics_new.compute_state_value()
        action_value_new = dynamics_new.compute_action_value()
        pi_old = pi_new
        pi_new = update_policy(pi_new, action_value_new)
        advances = state_value_new - state_value_old
        n_it += 1

    print("Policy iteration converged. (iteration={}, eps={})\n".format(n_it, np.sum(advances)))

    return pi_new, state_value_new, action_value_new