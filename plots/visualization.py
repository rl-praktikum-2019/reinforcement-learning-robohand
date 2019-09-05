import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH=os.environ['HOME']+'/git_tree/reinforcement-learning-2019/data'
PLOT_PATH=DATA_PATH+'/plots/'
print(os.getcwd())
NUM_OF_EXPERIMENTS=20
REWARD_PLOT_PATH=PLOT_PATH+'q_learning_plot_step_reward_'+str(NUM_OF_EXPERIMENTS)+'_line.png'
CUM_REWARD_PLOT_PATH=PLOT_PATH+'q_learning_plot_cum_reward_'+str(NUM_OF_EXPERIMENTS)+'_line.png'
BOXPLOT_PATH=PLOT_PATH+'q_learning_boxplot_step_reward_'+str(NUM_OF_EXPERIMENTS)+'_line.png'
Q_LEARNER_RESULTS_PATH= pd.read_json(DATA_PATH+'/q_learner_'+str(NUM_OF_EXPERIMENTS)+'_experiments.json')


def plot_rewards(configurations, index, rewards):
    return 0

def plot_cum_reward(cum_reward):
    plt.figure(figsize=(15,15))
    plt.title("Crawling robot - Cumulative Reward")
    plt.xlabel("Steps")
    plt.ylabel("cumulative reward")
    for i, row in cum_reward.iterrows():
        plt.plot(row['cum_reward'],label=row['configuration'])
        plt.legend()
    plt.savefig(CUM_REWARD_PLOT_PATH)

def plot_reward(step_reward):
    plt.figure(figsize=(15,15))
    plt.title("Crawling robot - Rewards per step")
    plt.xlabel("Steps")
    plt.ylabel("Average reward per step")
    for i, row in step_reward.iterrows():
        plt.plot(row['step_reward'],label=row['configuration'])
        plt.legend()
    plt.savefig(REWARD_PLOT_PATH)

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
    plt.savefig(BOXPLOT_PATH)

def append_experiment(step_reward_box, configuration: str, step_rewards):
  for reward in step_rewards:
    row=[configuration, reward]
    step_reward_box.loc[len(step_reward_box)] = row
    return step_reward_box

def crawling_robot_plots():
    data = pd.io.json.json_normalize(Q_LEARNER_RESULTS_PATH.results)
    data.sort_index()
    data = data.reindex(['num_experiments','steps_per_episode','alpha','epsilon','gamma','step_reward','cum_reward'], axis=1)

    data['configuration']= data.apply(lambda x: str(x['num_experiments'])+'_'+str(x['steps_per_episode'])
            +'_'+str(round(x['alpha'],2))+'_'+str(round(x['epsilon'],2))+'_'+
            str(round(x['gamma'],2)), axis=1)
    data=data.sort_values(by=['configuration'])

    plot_cum_reward(data[['configuration','cum_reward']])
    step_reward=data[['configuration','step_reward']]
    plot_reward(step_reward)

    col_names =  ['configuration', 'step_reward']
    step_reward_box  = pd.DataFrame(columns = col_names)

    for i, row in step_reward.iterrows():
        step_reward_box=append_experiment(step_reward_box, row['configuration'],row['step_reward'])
    boxplot(step_reward_box)

#plot_rewards(["some"],[1,2],[1,0.5])

if __name__ == "__main__":
    crawling_robot_plots()
