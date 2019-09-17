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
    def __init__(self, env):
        super(ThrowEnvWrapper, self).__init__(env)
        self.desired_ball_velocity = np.array([0, 0, 1])
        self.max_velocity = 0
        self.max_height = 0
        self.target_height = 0.4

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs["observation"]

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        ball_center_z = self.sim.data.get_joint_qpos(BALL_JOINT_NAME)[2]
        if ball_center_z <= BALL_RADIUS:
            print("Failure. Ball was dropped -> Reset Environment")
            done = True
        if ball_center_z > self.target_height:
            print("Success. Ball reached target height -> Reset Environment")
            done = True

        return observation["observation"], self.reward(reward), done, info

    def reward(self, reward):
        ball_center_pos = self.sim.data.get_joint_qpos(BALL_JOINT_NAME)
        ball_center_z = ball_center_pos[2]
        ball_velp = self.sim.data.get_joint_qvel(BALL_JOINT_NAME)[:3]
        ball_center_vel_z = ball_velp[2]

        print(ball_center_z)
        if ball_center_z > self.target_height:
            print("Reached height.")
            return 1000

        # if ball_center_vel_z > self.max_velocity:
        #     velocity_reward = ball_center_vel_z * 100
        #     reward += velocity_reward
        #     print("New achieved max velocity:", ball_center_vel_z)
        #     self.max_velocity = ball_center_vel_z

        # if ball_center_z > self.max_height:
        #     height_reward = ball_center_z * 100
        #     reward += height_reward
        #     self.max_height = ball_center_z
        #     print("New achieved max height:", ball_center_z)

        # delta_vel = self.desired_ball_velocity - ball_velp
        # d_vel = np.linalg.norm(delta_vel, axis=-1)
        # vel_reward = (10. * d_vel)
        # print("Velocity reward:", -vel_reward)
        # reward += -vel_reward

        if ball_center_z <= BALL_RADIUS:
            print("Ball was dropped: -20 reward")
            return -1000
        # print("Step-Reward:", reward)
        return ball_center_z - self.target_height * 10

    def get_ball_data(self):
        # Positional velocity
        x_pos = self.env.env.sim.data.get_body_xpos(BALL_OBJECT_NAME)

        # Positional velocity
        x_velp = self.env.env.sim.data.get_body_xvelp(BALL_OBJECT_NAME)

        # Rotational velocity
        x_velr = self.env.env.sim.data.get_body_xvelr(BALL_OBJECT_NAME)
        return x_pos, x_velp, x_velr
