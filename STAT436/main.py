from grid_world.grid_world import *
from grid_world.dynamics import *
from dp.policy_iteration import *
from dp.value_iteration import *
import time

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
    state_value = init_pi_dynamics.compute_state_value(exact=True)
    action_value = init_pi_dynamics.compute_action_value(exact=True)

    # run random action
    run_grid_world(pi, state_value, action_value)

    # policy iteration
    print("\nUpdating Policy via Policy Iteration")
    start_time = time.time()
    pi_new, state_value_new, action_value_new = policy_iteration(pi, gamma, reward, init_dynamics)
    end_time = time.time()
    computation_time = end_time - start_time
    print("Wall-clock time for Policy Iteration: {} sec\n".format(np.round(computation_time, 4)))

    # run updated policy
    run_grid_world(pi_new, state_value_new, action_value_new)
    
    # value iteration
    print("\nUpdating Policy via Value Iteration")
    start_time = time.time()
    init_action_value = np.zeros_like(action_value)
    optimal_action_value = value_iteration(init_action_value, gamma, reward, init_dynamics, eps=1e-3)
    end_time = time.time()
    computation_time = end_time - start_time
    print("Wall-clock time for Value Iteration: {} sec\n".format(np.round(computation_time, 4)))

    pi = np.array([0.25, 0.25, 0.25, 0.25]) #up, left, right, down
    pi = np.reshape(np.tile(pi, 12), (12, 4))
    pi_optimal = update_policy(pi, optimal_action_value)
    optimal_state_value = np.zeros((12, 1))
    optimal_state_value = np.max(optimal_action_value, axis=1)

    run_grid_world(pi_optimal, optimal_state_value, optimal_action_value)

    
    
    


    
