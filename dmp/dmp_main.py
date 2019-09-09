import numpy as np
import gym
import time
import pydmps.dmp_discrete

episodes = 2
episode_length = 2500
PLOT_FREQUENCY = 200

env = gym.make("ThrowBall-v0")

obs = env.reset()

trajectory = [[0, 0, 0], [0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, -.5], [0, 0, 0], [0, 0, .9]]
y_des = np.array(trajectory).T
y_des -= y_des[:, 0][:, None]

dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=3, n_bfs=100)
y_des = dmp.imitate_path(y_des=y_des)
action = np.zeros(20)

for j in range(episode_length):
    y_track, dy_track, ddy_track = dmp.step()
    action[1] = dy_track[2]
    print(y_track[2], dy_track[2], ddy_track[2])
    obs, reward, done, info = env.step(action)

    env.render()

env.close()
