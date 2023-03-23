from grid_world.grid_world import *
from grid_world.dynamics import *
from dp.policy_iteration import *

if __name__ == "__main__":
    gamma = 0.99
    # policy function
    pi = np.array([0.25, 0.25, 0.25, 0.25]) #up, left, right, down
    pi = np.reshape(np.tile(pi, 12), (12, 4))
    # reward
    reward = np.array([1, 0 ,-1])
    # initial dynamics with randomAgent
    
    init_dynamics = dynamics
    init_pi_dynamics = pi_dynamics(pi, gamma, reward, init_dynamics)
    state_value = init_pi_dynamics.compute_state_value()
    action_value = init_pi_dynamics.compute_action_value()

    # run random action
    run_grid_world(pi, state_value, action_value)

    # update policy via value function
    print("Updating Policy...")
    pi_new, state_value_new, action_value_new = policy_iteration(pi, gamma, reward, init_dynamics)
    
    # run updated policy
    run_grid_world(pi_new, state_value_new, action_value_new)
    optimal_action = policy_to_action(pi_new)
    
    
    
    


    
