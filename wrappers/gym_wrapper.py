import gym
import numpy as np

BALL_RADIUS = 0.03

BALL_OBJECT_NAME = "object"
BALL_JOINT_NAME = "object:joint"


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

        ball_center_z = self.sim.data.get_joint_qpos(BALL_JOINT_NAME)[2]
        if ball_center_z <= BALL_RADIUS:
            print("Ball was dropped -> Reset Environment")
            done = True

        return observation["observation"], self.reward(reward), done, info

    def reward(self, reward):
        ball_center_z = self.sim.data.get_joint_qpos(BALL_JOINT_NAME)[2]
        ball_center_vel_z = self.sim.data.get_joint_qvel(BALL_JOINT_NAME)[2]
        reward += ball_center_z * 10
        reward += ball_center_vel_z * 20

        if ball_center_z <= BALL_RADIUS:
            print("Ball was dropped: -20 reward")
            return -20
        print(reward)
        return reward

    def get_ball_data(self):
        # Positional velocity
        x_pos = self.env.env.sim.data.get_body_xpos(BALL_OBJECT_NAME)

        # Positional velocity
        x_velp = self.env.env.sim.data.get_body_xvelp(BALL_OBJECT_NAME)

        # Rotational velocity
        x_velr = self.env.env.sim.data.get_body_xvelr(BALL_OBJECT_NAME)
        return x_pos, x_velp, x_velr