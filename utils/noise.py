import numpy as np

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

#TODO: create a DMP noise to explore like described in DDPG paper
class DMPActionNoise:
    def __init__(self, dmp):
        self.dmp = dmp

    def __call__(self):
        y_track, dy_track, ddy_track = self.dmp.y, self.dmp.dy, self.dmp.ddy
        # Reduce the effect of dmp trajectory for other joints (fingers)
        clipped_ddy = np.clip(ddy_track, -0.8, 0)
        target = np.full((20,), clipped_ddy[0])
        # Remove action for horizontal wrist joint
        target[0] = 0
        # Use dmp attraction forces for vertical wrist joint
        target[1] = ddy_track[0]
        target * np.random.normal(size=self.target.shape)

        return target

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)