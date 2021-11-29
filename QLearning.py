import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import hiive.mdptoolbox 
import hiive.mdptoolbox.mdp
import hiive.mdptoolbox.example
import mdptoolbox, mdptoolbox.example
import gym
import matplotlib.pyplot as plt
import time
from gym.envs.toy_text.frozen_lake import generate_random_map

def plot_simple_data(x_var, y_var, x_label, y_label, title, figure_size=(6,4)):
    plt.figure(title)
    plt.rcParams["figure.figsize"] = figure_size
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.plot(x_var, y_var)#, 'o-')
    # plt.show()

def plot_data_legend(x_vars, x_label, all_y_vars, y_var_labels, y_label, title, y_bounds=None):
    plt.figure(title)
    colors = ['red','orange','black','green','blue','violet']
    plt.rcParams["figure.figsize"] = (6,4)

    i = 0
    for y_var in all_y_vars:

        plt.plot(x_vars, y_var,color=colors[i % 6], label=y_var_labels[i])#, 'o-', )
        i += 1
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if y_bounds != None:
        plt.ylim(y_bounds)
    leg = plt.legend()
    plt.savefig(title)
    # plt.show()


def make_time_array(run_stats, variables):
    cumulative_sum = 0
    times = []
    output_dict = {v:[] for v in variables}
    output_dict["times"] = times
    for result in run_stats:
        times.append(result["Time"])
        for v in result:
            if v in variables:
                output_dict[v].append(result[v])
    return output_dict
    

"""FROZEN LAKE"""


def run_frozen_lake(N = 8):
    """Plot Single iteration with multiple Gammas"""
    gammas = [0.15,0.25,0.5,0.75,0.85,0.95]#list(np.linspace(0.01,0.99,10))
    max_v = []
    mean_v = []
    time_ = []
    for gamma in gammas:
        
        random_map = generate_random_map(size=N, p=0.9)
        P, R = hiive.mdptoolbox.example.openai("FrozenLake-v1", desc=random_map)
        # P, R = hiive.mdptoolbox.example.forest(S=4, p=0.1)
        st = time.time()
        fm_q_mdp = hiive.mdptoolbox.mdp.QLearning(P, R, gamma, epsilon=0.1,epsilon_decay=0.5, epsilon_min=1e-6, n_iter=20000, alpha=0.1, alpha_decay=0.999, alpha_min = 1e-4,skip_check=False)
        fm_q_mdp.run()
        end = time.time()
        end-st
        fm_q_mdp.policy

        fm_q_curated_results = make_time_array(fm_q_mdp.run_stats, ["Mean V", "Max V", "Iteration"])
        print(fm_q_mdp.run_stats[-1])
        num_iters = len(fm_q_curated_results["Mean V"])

        mean_v.append(fm_q_curated_results["Mean V"])
        max_v.append(fm_q_curated_results["Max V"])
        time_.append(fm_q_curated_results["times"])

    plot_data_legend(fm_q_curated_results["Iteration"], "iteration", mean_v, gammas, "Mean Value", "Frozen Lake {}x{} - Q-Learning - Mean Value over Iteration".format(N,N), y_bounds=None)      

    plot_data_legend(fm_q_curated_results["Iteration"], "iteration", max_v, gammas, "Max Value (Total Reward)", "Frozen Lake {}x{} - Q-Learning - Max Value over Iteration".format(N,N), y_bounds=None)  

    plot_data_legend(fm_q_curated_results["Iteration"], "iteration", time_, gammas, "Time", "Frozen Lake {}x{} - Q-Learning - Elapsed Time over Iteration".format(N,N), y_bounds=None)  



    """ gamma vs total reward """

    gammas = list(np.linspace(0.01,0.99,30))
    max_v = []
    mean_v = []
    time_ = []
    for gamma in gammas:
        random_map = generate_random_map(size=N, p=0.9)
        P, R = hiive.mdptoolbox.example.openai("FrozenLake-v1", desc=random_map)
        # P, R = hiive.mdptoolbox.example.forest(S=4, p=0.1)
        st = time.time()
        fm_q_mdp = hiive.mdptoolbox.mdp.QLearning(P, R, gamma, epsilon=0.1,epsilon_decay=0.5, epsilon_min=1e-6, n_iter=20000, alpha=0.1, alpha_decay=0.999, alpha_min = 1e-4,skip_check=False)
        fm_q_mdp.run()
        end = time.time()
        end-st
        fm_q_mdp.policy

        fm_q_curated_results = make_time_array(fm_q_mdp.run_stats, ["Mean V", "Max V", "Iteration"])
        print(fm_q_mdp.run_stats[-1])
        num_iters = len(fm_q_curated_results["Mean V"])

        mean_v.append(fm_q_curated_results["Mean V"][-1])
        max_v.append(fm_q_curated_results["Max V"][-1])
        time_.append(fm_q_curated_results["times"][-1])

    plot_data_legend(gammas, "gamma", [mean_v,max_v], ["Mean Value", "Max Value"], "Value", "Frozen Lake {}x{} - Q-Learning - Value over Discount Factor".format(N,N), y_bounds=None)      
    plot_simple_data(gammas,mean_v,"gamma","Mean Value","Frozen Lake {}x{} - Q-Learning - Mean Value over Gamma".format(N,N))
    plot_simple_data(gammas,max_v,"gamma","Max Value","Frozen Lake {}x{} - Q-Learning - Mean Value over Gamma".format(N,N))

    """alpha vs total reward"""

    alphas = list(np.linspace(0.01,0.99,40))
    max_v = []
    mean_v = []
    time_ = []
    for alpha in alphas:
        random_map = generate_random_map(size=N, p=0.9)
        P, R = hiive.mdptoolbox.example.openai("FrozenLake-v1", desc=random_map)
        # P, R = hiive.mdptoolbox.example.forest(S=4, p=0.1)
        st = time.time()
        fm_q_mdp = hiive.mdptoolbox.mdp.QLearning(P, R, 0.95, epsilon=alpha,epsilon_decay=0.5, epsilon_min=1e-6, n_iter=20000, alpha=0.1, alpha_decay=0.999, alpha_min = 1e-4,skip_check=False)
        fm_q_mdp.run()
        end = time.time()
        end-st
        fm_q_mdp.policy

        fm_q_curated_results = make_time_array(fm_q_mdp.run_stats, ["Mean V", "Max V", "Iteration"])
        print(fm_q_mdp.run_stats[-1])
        num_iters = len(fm_q_curated_results["Mean V"])

        mean_v.append(fm_q_curated_results["Mean V"][-1])
        max_v.append(fm_q_curated_results["Max V"][-1])
        time_.append(fm_q_curated_results["times"][-1])

    plot_data_legend(alphas, "alpha", [mean_v,max_v], ["Mean Value", "Max Value"], "Value", "Frozen Lake {}x{} - Q-Learning - Value over Alpha".format(N,N), y_bounds=None)     
    plot_simple_data(alphas,mean_v,"alpha","Mean Value","Frozen Lake {}x{} - Q-Learning - Mean Value over Alpha".format(N,N))
    plot_simple_data(alphas,max_v,"alpha","Max Value","Frozen Lake {}x{} - Q-Learning - Max Value over Alpha".format(N,N))


    ''' epsilon vs total reward '''
    epsilons = list(np.linspace(0.01,0.99,40))
    max_v = []
    mean_v = []
    time_ = []
    for epsilon in epsilons:
        random_map = generate_random_map(size=N, p=0.9)
        P, R = hiive.mdptoolbox.example.openai("FrozenLake-v1", desc=random_map)
        # P, R = hiive.mdptoolbox.example.forest(S=4, p=0.1)
        st = time.time()
        fm_q_mdp = hiive.mdptoolbox.mdp.QLearning(P, R, 0.95, epsilon=epsilon,epsilon_decay=0.5, epsilon_min=1e-6, n_iter=20000, alpha=0.1, alpha_decay=0.999, alpha_min = 1e-4,skip_check=False)
        fm_q_mdp.run()
        end = time.time()
        end-st
        fm_q_mdp.policy
        fm_q_curated_results = make_time_array(fm_q_mdp.run_stats, ["Mean V", "Max V", "Iteration"])
        print(fm_q_mdp.run_stats[-1])
        num_iters = len(fm_q_curated_results["Mean V"])

        mean_v.append(fm_q_curated_results["Mean V"][-1])
        max_v.append(fm_q_curated_results["Max V"][-1])
        time_.append(fm_q_curated_results["times"][-1])

    plot_data_legend(epsilons, "epsilon", [mean_v,max_v], ["Mean Value", "Max Value"], "Value", "Frozen Lake {}x{} - Q-Learning - Value over Epsilon".format(N,N), y_bounds=None)      
    plot_simple_data(epsilons,mean_v,"epsilon","Mean Value","Frozen Lake {}x{} - Q-Learning - Mean Value over Epsilon".format(N,N))
    plot_simple_data(epsilons,max_v,"epsilon","Max Value","Frozen Lake {}x{} - Q-Learning - Max Value over Epsilon".format(N,N))

    plt.show()




"""FOREST MANAGEMENT"""
def run_forest(N = 4):
    """Plot Single iteration with single Gamma"""
    P, R = hiive.mdptoolbox.example.forest(S=N, p=0.1)
    st = time.time()
    fm_q_mdp = hiive.mdptoolbox.mdp.QLearning(P, R, 0.9, epsilon=0.1,epsilon_decay=0.5, epsilon_min=1e-6, n_iter=20000, alpha=0.1, alpha_decay=0.999, alpha_min = 1e-4,skip_check=False)
    fm_q_mdp.run()
    end = time.time()
    end-st
    fm_q_mdp.policy

    fm_q_curated_results = make_time_array(fm_q_mdp.run_stats, ["Mean V", "Max V", "Iteration"])
    print(fm_q_mdp.run_stats[-1])
    num_iters = len(fm_q_curated_results["Mean V"])
    plot_simple_data(fm_q_curated_results["Iteration"], fm_q_curated_results["Mean V"], 
                     "iteration", "Mean Value", "Forest (64 states) - Q-Learning - Mean Value over Iteration", figure_size=(6,4))
    plot_simple_data(fm_q_curated_results["Iteration"], fm_q_curated_results["Max V"], 
                     "iteration", "Max Value (total reward)", "Forest (64 states) - Q-Learning - Max Value (Reward) over Iteration", figure_size=(6,4))
    plot_simple_data(fm_q_curated_results["Iteration"], fm_q_curated_results["times"], 
                     "iteration", "time elapsed (seconds)", "Forest (64 states) - Q-Learning - Time Elapsed over Iteration", figure_size=(6,4))

    """Plot Single iteration with multiple Gammas"""
    gammas = [0.15,0.25,0.5,0.75,0.85,0.95]#list(np.linspace(0.01,0.99,10))
    max_v = []
    mean_v = []
    time_ = []
    for gamma in gammas:
        P, R = hiive.mdptoolbox.example.forest(S=4, p=0.1)
        st = time.time()
        fm_q_mdp = hiive.mdptoolbox.mdp.QLearning(P, R, gamma, epsilon=0.1,epsilon_decay=0.5, epsilon_min=1e-6, n_iter=20000, alpha=0.1, alpha_decay=0.999, alpha_min = 1e-4,skip_check=False)
        fm_q_mdp.run()
        end = time.time()
        end-st
        fm_q_mdp.policy



        fm_q_curated_results = make_time_array(fm_q_mdp.run_stats, ["Mean V", "Max V", "Iteration"])
        print(fm_q_mdp.run_stats[-1])
        num_iters = len(fm_q_curated_results["Mean V"])

        mean_v.append(fm_q_curated_results["Mean V"])
        max_v.append(fm_q_curated_results["Max V"])
        time_.append(fm_q_curated_results["times"])

    plot_data_legend(fm_q_curated_results["Iteration"], "iteration", mean_v, gammas, "Mean Value", "Forest (4 states) - Q-Learning - Mean Value over Iteration", y_bounds=None)      

    plot_data_legend(fm_q_curated_results["Iteration"], "iteration", max_v, gammas, "Max Value (Total Reward)", "Forest (4 states) - Q-Learning - Max Value over Iteration", y_bounds=None)  

    plot_data_legend(fm_q_curated_results["Iteration"], "iteration", time_, gammas, "Time", "Forest (4 states) - Q-Learning - Elapsed Time over Iteration", y_bounds=None)  



    ''' gamma vs total reward '''

    gammas = list(np.linspace(0.01,0.99,30))
    max_v = []
    mean_v = []
    time_ = []
    for gamma in gammas:
        P, R = hiive.mdptoolbox.example.forest(S=4, p=0.1)
        st = time.time()
        fm_q_mdp = hiive.mdptoolbox.mdp.QLearning(P, R, gamma, epsilon=0.1,epsilon_decay=0.5, epsilon_min=1e-6, n_iter=20000, alpha=0.1, alpha_decay=0.999, alpha_min = 1e-4,skip_check=False)
        fm_q_mdp.run()
        end = time.time()
        end-st
        fm_q_mdp.policy

        fm_q_curated_results = make_time_array(fm_q_mdp.run_stats, ["Mean V", "Max V", "Iteration"])
        print(fm_q_mdp.run_stats[-1])
        num_iters = len(fm_q_curated_results["Mean V"])

        mean_v.append(fm_q_curated_results["Mean V"][-1])
        max_v.append(fm_q_curated_results["Max V"][-1])
        time_.append(fm_q_curated_results["times"][-1])

    plot_data_legend(gammas, "gamma", [mean_v,max_v], ["Mean Value", "Max Value"], "Value", "Forest (4 states) - Q-Learning - Value over Discount Factor", y_bounds=None)      


    """alpha vs total reward"""

    alphas = list(np.linspace(0.01,0.99,40))
    max_v = []
    mean_v = []
    time_ = []
    for alpha in alphas:
        P, R = hiive.mdptoolbox.example.forest(S=4, p=0.1)
        st = time.time()
        fm_q_mdp = hiive.mdptoolbox.mdp.QLearning(P, R, 0.95, epsilon=alpha,epsilon_decay=0.5, epsilon_min=1e-6, n_iter=20000, alpha=0.1, alpha_decay=0.999, alpha_min = 1e-4,skip_check=False)
        fm_q_mdp.run()
        end = time.time()
        end-st
        fm_q_mdp.policy

        fm_q_curated_results = make_time_array(fm_q_mdp.run_stats, ["Mean V", "Max V", "Iteration"])
        print(fm_q_mdp.run_stats[-1])
        num_iters = len(fm_q_curated_results["Mean V"])

        mean_v.append(fm_q_curated_results["Mean V"][-1])
        max_v.append(fm_q_curated_results["Max V"][-1])
        time_.append(fm_q_curated_results["times"][-1])

    plot_data_legend(alphas, "alpha", [mean_v,max_v], ["Mean Value", "Max Value"], "Value", "Forest (4 states) - Q-Learning - Value over Alpha", y_bounds=None)     

    ''' epsilon vs total reward '''
    epsilons = list(np.linspace(0.01,0.99,40))
    max_v = []
    mean_v = []
    time_ = []
    for epsilon in epsilons:
        P, R = hiive.mdptoolbox.example.forest(S=4, p=0.1)
        st = time.time()
        fm_q_mdp = hiive.mdptoolbox.mdp.QLearning(P, R, 0.95, epsilon=epsilon,epsilon_decay=0.5, epsilon_min=1e-6, n_iter=20000, alpha=0.1, alpha_decay=0.999, alpha_min = 1e-4,skip_check=False)
        fm_q_mdp.run()
        end = time.time()
        end-st
        fm_q_mdp.policy
        fm_q_curated_results = make_time_array(fm_q_mdp.run_stats, ["Mean V", "Max V", "Iteration"])
        print(fm_q_mdp.run_stats[-1])
        num_iters = len(fm_q_curated_results["Mean V"])

        mean_v.append(fm_q_curated_results["Mean V"][-1])
        max_v.append(fm_q_curated_results["Max V"][-1])
        time_.append(fm_q_curated_results["times"][-1])

    plot_data_legend(epsilons, "epsilon", [mean_v,max_v], ["Mean Value", "Max Value"], "Value", "Forest (4 states) - Q-Learning - Value over Epsilon", y_bounds=None)      
    plt.show()


if __name__ == "__main__":
    run_frozen_lake(N = 8)
    run_forest(N = 4)
