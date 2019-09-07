import os,time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import sleep

# TODO: Improve naming schema
# TODO: Integrate visualization to ddpg_main.py & others.
# XXX: Do we want to show several plots live?
# XXX: Do we need multiple lines and labels for update_plot()? 
#       A: I think not since multiple lines are only for presentation &
#       update_plot() is a live demo method.

DATA_PATH=os.path.dirname(os.path.realpath(__file__))+'/../data'
PLOT_PATH=DATA_PATH+'/plots/'
PNG='.png'
REWARD_PLOT_PATH=PLOT_PATH+'reward_'
CUM_REWARD_PLOT_PATH=PLOT_PATH+'cum_reward_'
BOXPLOT_PATH=PLOT_PATH+'boxplot_reward_'
Q_LEARNER_RESULTS_PATH= DATA_PATH+'/q_learner_20_results.json'

plt.ion()
#plt.figure(figsize=(20, 10))

# TODO: This method is not being used.
def close():
    ## disable interactive plotting => otherwise window terminates
    plt.ioff()
    plt.show()

# TODO: Add labels to ax in update_plot() plots
# TODO: Test if update_plot() also works for boxplot
# FIXME: X axis does not show the number of steps correctly
def update_plot(plt, label, data):
    ax1 = plt.gca()
    ax1.clear()
    plt.title(plt.title)
    plt.xlabel("Steps")
    plt.ylabel("Average reward per step")
    ax1.plot(data,label=label)
    plt.show()
    plt.pause(.001) # Prevents mujoco from blocking rendering


def results_preprocessing(configuration, rewards, cum_rewards):
    data = pd.DataFrame(
    {'configuration': [configuration,],
     'step_reward': [rewards,],
     'cum_reward': [cum_rewards,]
    })
    print(data)
    return data

def plot_cum_reward(cum_reward):
    fig=plt.figure()
    plt.title("Cumulative Reward - Plot")
    plt.xlabel("Steps")
    plt.ylabel("cumulative reward")
    ax1 = fig.add_subplot(1,1,1)
    for i, row in cum_reward.iterrows():
        ax1.plot(row['cum_reward'],label=row['configuration'])
        plt.legend()
    return plt

def plot_reward(step_reward):
    plt.figure(figsize=(15,15))
    plt.title("Rewards per step - Plot")
    plt.xlabel("Steps")
    plt.ylabel("Average reward per step")
    for i, row in step_reward.iterrows():
        plt.plot(row['step_reward'],label=row['configuration'])
        plt.legend()
    return plt

def boxplot(step_reward_box):
    plt.figure(figsize=(20,10))
    plt.xticks(rotation=-45)
    # plot boxplot with seaborn
    bplot=sns.boxplot(y='step_reward', x='configuration',
                    data=step_reward_box,
                    width=0.75,
                    palette="colorblind")
    # add swarmplot
    bplot=sns.swarmplot(y='step_reward', x='configuration',
                data=step_reward_box,
                color='black',
                alpha=0.75)
    return plt

def append_experiment(step_reward_box, configuration: str, step_rewards):
  for reward in step_rewards:
    row=[configuration, reward]
    step_reward_box.loc[len(step_reward_box)] = row
    return step_reward_box

def plot_dynamic(learner_name, data):
    cum_reward_plot=plot_cum_reward(data[['configuration','cum_reward']])
    #cum_reward_plot.show()
    #cum_reward_plot.savefig(CUM_REWARD_PLOT_PATH+learner_name+PNG)
    return cum_reward_plot

def plot_everything(learner_name, data):
    cum_reward_plot=plot_cum_reward(data[['configuration','cum_reward']])
    cum_reward_plot.savefig(CUM_REWARD_PLOT_PATH+learner_name+PNG)

    step_reward=data[['configuration','step_reward']]
    reward_plot=plot_reward(step_reward)
    reward_plot.savefig(REWARD_PLOT_PATH+learner_name+PNG)

    col_names =  ['configuration', 'step_reward']
    step_reward_box  = pd.DataFrame(columns = col_names)

    for i, row in step_reward.iterrows():
        step_reward_box=append_experiment(step_reward_box, row['configuration'],row['step_reward'])
    box_plot=boxplot(step_reward_box)
    box_plot.savefig(BOXPLOT_PATH+learner_name+PNG)

def random_robby_plots(configuration,rewards,cum_rewards):
    data = results_preprocessing(configuration, rewards, cum_rewards)
    cum_reward_plot=plot_dynamic('random_robby', data)
    #plot_everything('random_robby', data)
    return cum_reward_plot

def crawling_robot_plots():
    q_results= pd.read_json(Q_LEARNER_RESULTS_PATH)
    data = pd.io.json.json_normalize(q_results.results)

    data.sort_index()
    data = data.reindex(['num_experiments','steps_per_episode','alpha','epsilon','gamma','step_reward','cum_reward'], axis=1)
    data['configuration']= data.apply(lambda x: str(x['num_experiments'])+'_'+str(x['steps_per_episode'])
            +'_'+str(round(x['alpha'],2))+'_'+str(round(x['epsilon'],2))+'_'+
            str(round(x['gamma'],2)), axis=1)
    data=data.sort_values(by=['configuration'])
    data=data[['configuration','step_reward','cum_reward']]
    data = results_preprocessing(data)
    plot_everything('q_learner_20', data)

#plot_rewards(["some"],[1,2],[1,0.5])

if __name__ == "__main__":
    crawling_robot_plots()
