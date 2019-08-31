import gym


class ObservationWrapper(gym.ObservationWrapper):
    #
    #
    # Adapt observation # 1 - We only need the pure observations no goals
    #
    #
    def __init__(self, env):
        super(ObservationWrapper, self).__init__(env)

    def observation(self, observation):
        return observation["observation"]