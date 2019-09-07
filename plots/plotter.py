import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

# TODO: Delete this file when its knowledge is no longer needed. 

def initialize():
    ## interactive plotting on (no need to close window for next iteration)
    plt.ion()
    plt.figure(figsize=(20, 10))

def close():
    ## disable interactive plotting => otherwise window terminates
    plt.ioff()
    plt.show()

def plot(reward_memory, current_step, method_name, epsilon, num_plays):
    ci = 0.95  # 95% confidence interval
    means = np.mean(reward_memory, axis=0)
    stds = np.std(reward_memory, axis=0)
    n = means.size

    # compute upper/lower confidence bounds
    test_stat = st.t.ppf((ci + 1) / 2, e)
    lower_bound = means - test_stat * stds / np.sqrt(current_step)
    upper_bound = means + test_stat * stds / np.sqrt(current_step)

    this_manager = plt.get_current_fig_manager()
    this_manager.window.wm_geometry("+0+0")

    # clear plot frame
    plt.clf()

    # plot average reward
    plt.plot(means, color='blue', label="epsilon=%.2f" % epsilon)

    # plot upper/lower confidence bound
    x = np.arange(0, num_plays, 1)
    plt.fill_between(x=x, y1=lower_bound, y2=upper_bound, color='blue', alpha=0.2, label="CI %.2f" % ci)

    plt.grid()
    plt.ylim(0, 2)  # limit y axis
    plt.title(method_name + ': Avg. Reward per step in experiment %d: %.4f' % (current_step, sum(means) / num_plays))
    plt.ylabel("Reward per step")
    plt.xlabel("Play")
    plt.legend()
    plt.show()
    plt.pause(0.01)