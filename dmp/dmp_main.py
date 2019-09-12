import numpy as np
import gym
import pydmps.dmp_discrete
from wrappers.gym_wrapper import ThrowEnvWrapper
import math

episodes = 2
episode_length = 2500
PLOT_FREQUENCY = 200

env = ThrowEnvWrapper(gym.make('HandManipulateEgg-v0'))

def main(args=[]):
    obs = env.reset()

# 1-dimensional since joint can only move in one axis -> up/down axis
    trajectory = [[0], [0], [0], [0], [-1], [1]]
    y_des = np.array(trajectory).T
    y_des -= y_des[:, 0][:, None]

    dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=1, n_bfs=600)
    y_des = dmp.imitate_path(y_des=y_des)
    action = np.zeros(20)
    time_steps = dmp.timesteps
    print(time_steps)
    for j in range(time_steps):
        # What force to use? vel/acc or even position?
        y_track, dy_track, ddy_track = dmp.step()

        # All joints controlled by SAME dmp force
        dampening_term = 0.01 * 1.0/(1.0+j)
        action = np.full((20, ), ddy_track[0] * dampening_term)

        # Only wrist joint controlled by dmp force
        action[1] = ddy_track[0]

        # print(y_track, dy_track, ddy_track)
        obs, reward, done, info = env.step(action)
        print("Reward:", reward)
        if done:
            break
        env.render()

    env.close()

main()
