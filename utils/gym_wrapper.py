import gym
import numpy as np

BALL_RADIUS = 0.03

BALL_OBJECT_NAME = "object"
BALL_JOINT_NAME = "object:joint"


class ThrowEnvWrapper(gym.Wrapper):
    #
    #
    # Gym wrapper with features:
    # - new reward function
    # - detect ball collision with ground and reset
    #
    #
    def __init__(self, env):
        super(ThrowEnvWrapper, self).__init__(env)
        self.desired_ball_velocity = np.array([0, 0, 1])
        self.max_velocity = 0
        self.max_height = 0
        self.target_height = 0.4
        self.ball_velp = np.zeros((3,))
        self.ball_center_z = 0
        self.ball_center_vel_z = 0
        self.reached_target = False

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs["observation"]

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        ball_center_pos = self.sim.data.get_joint_qpos(BALL_JOINT_NAME)
        self.ball_center_z = ball_center_pos[2]
        self.ball_velp = self.sim.data.get_joint_qvel(BALL_JOINT_NAME)[:3]
        self.ball_center_vel_z = self.ball_velp[2]

        self.ball_center_z = self.sim.data.get_joint_qpos(BALL_JOINT_NAME)[2]
        if self.ball_center_z <= BALL_RADIUS:
            print("Ball was dropped -> Reset Environment")
            done = True

        return observation["observation"], self.reward(reward), done, info

    def reward(self, reward):
        # self.reward_functionB(reward)
        return self.reward_functionB()
        #return reward

    def reward_functionA(self):
        reward = 1 - (self.target_height - self.ball_center_z) / self.target_height
        z_direction = np.sign(self.ball_center_vel_z)
        if z_direction > 0:
            reward += self.ball_center_vel_z * 10
        return reward

    def reward_functionB(self):
        reward = 1 - (self.target_height - self.ball_center_z) / self.target_height
        z_direction = np.sign(self.ball_center_vel_z)

        if not self.reached_target and self.ball_center_z > self.target_height:
            reward += 100
            self.reached_target = True

        if z_direction < 0:
            reward = 0

        return reward
