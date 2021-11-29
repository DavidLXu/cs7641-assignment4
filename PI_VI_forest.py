# frozen-lake-ex1.py
# import gym # loading the Gym library

# import hiive.mdptoolbox as mdp

# from openai import OpenAI_MDPToolbox

# env = gym.make("FrozenLake-v1")
# # print(env.env.P)

# open_MDP = OpenAI_MDPToolbox("FrozenLake-v1")

# print(open_MDP.R)

# print(open_MDP.P)

import hiive.mdptoolbox.openai

import hiive.mdptoolbox.example
from gym.envs.toy_text.frozen_lake import generate_random_map

N = 4
# random_map = generate_random_map(size=N, p=0.9)
problem = "Forest ({} states) - ".format(N)

# P, R = hiive.mdptoolbox.example.openai("FrozenLake-v1", desc=random_map)


P, R = hiive.mdptoolbox.example.forest(S=N, r1=4, r2=2, p=0.1, is_sparse=False)#(S=S, p=0.1)


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import numpy as np




### Policy Iteration ###

logger.info("Policy Iteration")
pi = hiive.mdptoolbox.mdp.PolicyIteration(transitions = P,
                                          reward = R, gamma = 0.99, 
                                          policy0=None,
                                          max_iter=50, eval_type=0, skip_check=False,
                                          run_stat_frequency=None)
# pi.setVerbose()
pi.run()
# print(pi.policy)
# print(pi.run_stats)


# Plot iteration curve with same gamma
max_v = []
mean_v = []
reward = []
error = []
time = []
iteration = []

for i in range(len(pi.run_stats)):
    max_v.append(pi.run_stats[i]["Max V"])
    mean_v.append(pi.run_stats[i]["Mean V"])
    reward.append(pi.run_stats[i]["Reward"])
    error.append(pi.run_stats[i]["Error"])
    time.append(pi.run_stats[i]["Time"])
    iteration.append(pi.run_stats[i]["Iteration"])

algo_name = "Policy Iteration - "

print("max_v",max_v[-1])
print("mean_v",mean_v[-1])
print("reward",reward[-1])
print("time",time[-1])
print("iteration",iteration[-1])
print("error",error[-1])


plt.figure(problem+algo_name+"Mean Value Over Iteration")
plt.plot(iteration,mean_v)
plt.xlabel("iteration")
plt.ylabel("Mean Value")
plt.title(problem+algo_name+"Mean Value Over Iteration")
plt.savefig(problem+algo_name+"Mean Value Over Iteration")


plt.figure(problem+algo_name+"Max Value Over Iteration")
plt.plot(iteration,max_v)
plt.xlabel("iteration")
plt.ylabel("Max Value")
plt.title(problem+algo_name+"Max Value Over Iteration")
plt.savefig(problem+algo_name+"Max Value Over Iteration")

plt.figure(problem+algo_name+"Reward Over Iteration")
plt.plot(iteration,reward)
plt.xlabel("iteration")
plt.ylabel("Reward")
plt.title(problem+algo_name+"Reward Over Iteration")
plt.figure(problem+algo_name+"Reward Over Iteration")

plt.figure(problem + algo_name+"Elapsed Time Over Iteration")
plt.title(problem + algo_name+"Elapsed Time Over Iteration")
plt.plot(iteration,time)
plt.xlabel("iteration")
plt.ylabel("Time")
plt.savefig(problem + algo_name+"Elapsed Time Over Iteration")

# plt.show()


''' Plot curve with different gammas '''
max_v_last = []
mean_v_last = []
reward_last = []
gammas = [0.3,0.4,0.5,0.6,0.7,0.8,0.99]
for gamma in gammas:
    pi = hiive.mdptoolbox.mdp.PolicyIteration(transitions = P,
                                            reward = R, gamma = gamma, 
                                            policy0=None,
                                            max_iter=10, eval_type=0, skip_check=False,
                                            run_stat_frequency=None)
    #pi.setVerbose()
    pi.run()
    max_v_last.append(pi.run_stats[-1]["Max V"])
    mean_v_last.append(pi.run_stats[-1]["Mean V"])
    reward_last.append(pi.run_stats[-1]["Reward"])

plt.figure(problem + algo_name+"Max Value Over Different Discounts")
plt.title(problem + algo_name+"Max Value Over Different Discounts")
plt.plot(gammas,max_v_last)
plt.xlabel("Gamma")
plt.ylabel("Max Value")
plt.savefig(problem + algo_name+"Max Value Over Different Discounts")

plt.figure(problem + algo_name+"Mean Value Over Different Discounts")
plt.title(problem+algo_name+"Mean Value Over Different Discounts")
plt.plot(gammas,mean_v_last)
plt.xlabel("Gamma")
plt.ylabel("Mean Value")
plt.savefig(problem+algo_name+"Mean Value Over Different Discounts")

plt.figure(problem+algo_name+"Reward Over Different Discounts")
plt.title(problem+algo_name+"Reward Over Different Discounts")
plt.plot(gammas,reward_last)
plt.xlabel("Gamma")
plt.ylabel("Reward Value")
plt.savefig(problem+algo_name+"Reward Over Different Discounts")



'''Policy Iteration, iteration curve with different gamma'''

logger.info("Policy Iteration, iteration curve with different gamma")
algo_name = "Policy Iteration - "

max_v_all = []
mean_v_all = []
reward_all = []
error_all = []
time_all = []
iteration_all = []

gammas = [0.3,0.4,0.5,0.6,0.7,0.8,0.99]
for gamma in gammas:
    pi = hiive.mdptoolbox.mdp.PolicyIteration(transitions = P,
                                            reward = R, gamma = gamma, 
                                            policy0=None,
                                            max_iter=50, eval_type=0, skip_check=False,
                                            run_stat_frequency=None)
    pi.setVerbose()
    pi.run()
    # print(pi.policy)
    # print(pi.run_stats)

    max_v = []
    mean_v = []
    reward = []
    error = []
    time = []
    iteration = []

    for i in range(len(pi.run_stats)):
        max_v.append(pi.run_stats[i]["Max V"])
        mean_v.append(pi.run_stats[i]["Mean V"])
        reward.append(pi.run_stats[i]["Reward"])
        error.append(pi.run_stats[i]["Error"])
        time.append(pi.run_stats[i]["Time"])
        iteration.append(pi.run_stats[i]["Iteration"])

    plt.figure(problem+algo_name+"Mean Value Over Iteration")
    plt.plot(iteration[:8],mean_v[:8])
    plt.xlabel("iteration")
    plt.ylabel("Mean Value")
    plt.legend([0.3,0.4,0.5,0.6,0.7,0.8,0.99])
    plt.title(problem+algo_name+"Mean Value Over Iteration")
    plt.savefig(problem+algo_name+"Mean Value Over Iteration")

    plt.figure(problem+algo_name+"Max Value (Reward) Over Iteration")
    plt.plot(iteration[:8],max_v[:8])
    plt.xlabel("iteration")
    plt.ylabel("Max Value")
    plt.legend([0.3,0.4,0.5,0.6,0.7,0.8,0.99])
    plt.title(problem+algo_name+"Max Value (Reward) Over Iteration")
    plt.savefig(problem+algo_name+"Max Value (Reward) Over Iteration")

    plt.figure(problem+algo_name+"Reward Over Iteration")
    plt.plot(iteration[:8],reward[:8])
    plt.xlabel("iteration")
    plt.ylabel("Reward")
    plt.legend([0.3,0.4,0.5,0.6,0.7,0.8,0.99])
    plt.title(problem+algo_name+"Reward Over Iteration")
    plt.figure(problem+algo_name+"Reward Over Iteration")

    plt.figure(problem + algo_name+"Elapsed Time Over Iteration")
    plt.title(problem + algo_name+"Elapsed Time Over Iteration")
    plt.plot(iteration[:8],time[:8])
    plt.xlabel("iteration")
    plt.ylabel("Time")
    plt.legend([0.3,0.4,0.5,0.6,0.7,0.8,0.99])
    plt.savefig(problem + algo_name+"Elapsed Time Over Iteration")
    


# ### Value Iteration ###


logger.info("Value Iteration, iteration curve with different gamma")
algo_name = "Value Iteration - "

max_v_all = []
mean_v_all = []
reward_all = []
error_all = []
time_all = []
iteration_all = []

gammas = [0.3,0.4,0.5,0.6,0.7,0.8,0.99]
for gamma in gammas:
    vi = hiive.mdptoolbox.mdp.ValueIteration(transitions = P,
                                            reward = R, gamma = gamma, 
                                            epsilon=1e-2,
                                            max_iter=1000, 
                                            initial_value=0, 
                                            skip_check=False,
                                            run_stat_frequency=None)
    vi.setVerbose()
    vi.run()
    # print(pi.policy)
    # print(pi.run_stats)

    max_v = []
    mean_v = []
    reward = []
    error = []
    time = []
    iteration = []

    for i in range(len(vi.run_stats)):
        max_v.append(vi.run_stats[i]["Max V"])
        mean_v.append(vi.run_stats[i]["Mean V"])
        reward.append(vi.run_stats[i]["Reward"])
        error.append(vi.run_stats[i]["Error"])
        time.append(vi.run_stats[i]["Time"])
        iteration.append(vi.run_stats[i]["Iteration"])




    plt.figure(problem+algo_name+"Mean Value Over Iteration")
    plt.plot(iteration[:30],mean_v[:30])
    plt.xlabel("iteration")
    plt.ylabel("Mean Value")
    plt.legend([0.3,0.4,0.5,0.6,0.7,0.8,0.99])
    plt.title(problem+algo_name+"Mean Value Over Iteration")
    plt.savefig(problem+algo_name+"Mean Value Over Iteration")


    plt.figure(problem+algo_name+"Max Value (Reward) Over Iteration")
    plt.plot(iteration[:30],max_v[:30])
    plt.xlabel("iteration")
    plt.ylabel("Max Value")
    plt.legend([0.3,0.4,0.5,0.6,0.7,0.8,0.99])
    plt.title(problem+algo_name+"Max Value (Reward) Over Iteration")
    plt.savefig(problem+algo_name+"Max Value (Reward) Over Iteration")

    plt.figure(problem+algo_name+"Reward Over Iteration")
    plt.plot(iteration[:30],reward[:30])
    plt.xlabel("iteration")
    plt.ylabel("Reward")
    plt.legend([0.3,0.4,0.5,0.6,0.7,0.8,0.99])
    plt.title(problem+algo_name+"Reward Over Iteration")
    plt.figure(problem+algo_name+"Reward Over Iteration")

    plt.figure(problem + algo_name+"Elapsed Time Over Iteration")
    plt.title(problem + algo_name+"Elapsed Time Over Iteration")
    plt.plot(iteration[:30],time[:30])
    plt.xlabel("iteration")
    plt.ylabel("Time")
    plt.legend([0.3,0.4,0.5,0.6,0.7,0.8,0.99])
    plt.savefig(problem + algo_name+"Elapsed Time Over Iteration")




''' Plot gamma curve with different gammas '''

max_v_last = []
mean_v_last = []
reward_last = []
gammas = list(np.linspace(0.001,0.999,50))
for gamma in gammas:
    vi = hiive.mdptoolbox.mdp.ValueIteration(transitions = P,
                                            reward = R, gamma = gamma, 
                                            epsilon=0.01,
                                            max_iter=1000, 
                                            initial_value=0, 
                                            skip_check=False,
                                            run_stat_frequency=None)

    vi.run()
    max_v_last.append(vi.run_stats[-1]["Max V"])
    mean_v_last.append(vi.run_stats[-1]["Mean V"])
    reward_last.append(vi.run_stats[-1]["Reward"])

plt.figure(problem + algo_name+"Max Value Over Different Discounts")
plt.title(problem + algo_name+"Max Value Over Different Discounts")
plt.plot(gammas,max_v_last)
plt.xlabel("Gamma")
plt.ylabel("Max Value")
plt.savefig(problem + algo_name+"Max Value Over Different Discounts")

plt.figure(problem + algo_name+"Mean Value Over Different Discounts")
plt.title(problem + algo_name+"Mean Value Over Different Discounts")
plt.plot(gammas,mean_v_last)
plt.xlabel("Gamma")
plt.ylabel("Mean Value")
plt.savefig(problem + algo_name+"Mean Value Over Different Discounts")

plt.figure(problem + algo_name+"Reward Over Different Discounts")
plt.title(problem + algo_name+"Reward Over Different Discounts")
plt.plot(gammas,reward_last)
plt.xlabel("Gamma")
plt.ylabel("Reward Value")
plt.savefig(problem + algo_name+"Reward Over Different Discounts")

plt.show()

