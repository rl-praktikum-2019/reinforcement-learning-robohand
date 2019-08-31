import gym
import numpy as np

BALL_OBJECT_NAME = "object"


class ThrowEnvWrapper(gym.Wrapper):
    #
    #
    # Adapt gym - The final adapted gym wrapper with the best adaptions for reward and observations
    #
    #
    def __init__(self, env, desired_ball_velocity=np.array([0, 0, 1])):
        super(ThrowEnvWrapper, self).__init__(env)
        self.desired_ball_velocity = desired_ball_velocity

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs["observation"]

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation["observation"], self.reward(reward), done, info

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