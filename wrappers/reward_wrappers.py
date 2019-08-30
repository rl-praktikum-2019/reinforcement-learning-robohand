import gym
import numpy as np

BALL_OBJECT_NAME = "object"


class MaxHeightRewardWrapper(gym.RewardWrapper):
    #
    #
    # Adapt reward # 1 - Maximize height and velocity, reward height increase
    #
    #
    def __init__(self, env, epsilon=0.1):
        super(MaxHeightRewardWrapper, self).__init__(env)

        self.epsilon = epsilon
        self.prev_height = 0
        self.max_velocity = 0
        self.max_height = 0

    def reward(self, reward):
        p, vp, vr = self.get_ball_data()
        height = p[2]
        velocity_z = vp[2]

        if height > self.prev_height:
            reward += 2
            print("Increased height:", height)
            self.prev_height = height
        else:
            reward -= 2

        if velocity_z > self.max_velocity:
            velocity_reward = velocity_z * 1000
            reward += velocity_reward
            print("New achieved max velocity:", height)
            self.max_velocity = velocity_z

        if height > self.max_height:
            height_reward = height * 100
            reward += height_reward
            self.max_height = height
            print("New achieved max height:", height)

        return reward

    def get_ball_data(self):
        # Positional velocity
        x_pos = self.env.env.sim.data.get_body_xpos(BALL_OBJECT_NAME)
        # Positional velocity
        x_velp = self.env.env.sim.data.get_body_xvelp(BALL_OBJECT_NAME)
        # Rotational velocity
        x_velr = self.env.env.sim.data.get_body_xvelr(BALL_OBJECT_NAME)
        return x_pos, x_velp, x_velr


class VelocityRewardWrapper(gym.RewardWrapper):
    #
    #
    # Adapt reward # 2 - Achieve desired velocity vector
    #
    #
    def __init__(self, env, epsilon=0.1):
        super(VelocityRewardWrapper, self).__init__(env)

        self.epsilon = epsilon
        self.prev_height = 0
        self.max_velocity = 0
        self.max_height = 0
        self.desired_ball_velocity = np.array([0, 0, 1])

    def reward(self, reward):
        _, positional_velocity, _ = self.get_ball_data()
        delta_vel = self.desired_ball_velocity - positional_velocity
        d_vel = np.linalg.norm(delta_vel, axis=-1)
        vel_reward = (10. * d_vel)
        print("Velocity reward:", -vel_reward)
        reward += -vel_reward

        return reward

    def get_ball_data(self):
        # Positional velocity
        x_pos = self.env.env.sim.data.get_body_xpos(BALL_OBJECT_NAME)
        # Positional velocity
        x_velp = self.env.env.sim.data.get_body_xvelp(BALL_OBJECT_NAME)
        # Rotational velocity
        x_velr = self.env.env.sim.data.get_body_xvelr(BALL_OBJECT_NAME)
        return x_pos, x_velp, x_velr
