from grid_world.grid_world import *
from RL.mc import *
import time

if __name__ == "__main__":
    gamma = 0.99
    # random policy function
    pi = np.array([0.25, 0.25, 0.25, 0.25]) #up, left, right, down
    pi = np.reshape(np.tile(pi, 12), (12, 4))
    
    print("\nUpdating Policy via Policy Iteration w/ Monte-Carlo")
    start_time = time.time()
    pi_new, action_value_new = mc_policy_iteration(pi, Agent, gamma, play_num=100, epsilon=0.1)
    end_time = time.time()
    computation_time = end_time - start_time
    print("Wall-clock time for Policy Iteration: {} sec\n".format(np.round(computation_time, 4)))

    print("Let's run grid world!")
    agent = Agent(pi_new)
    success_rate = agent.play(100, stat=True)
    agent.show_policy()
    print("action value:\n {}".format(np.round(action_value_new, 3)))
    print("Success rate:{} %".format(success_rate * 100))
