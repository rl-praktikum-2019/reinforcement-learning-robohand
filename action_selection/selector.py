import numpy as np


class Selector():
    def __init__(self, n, eps, env):
        self.env = env
        self.n = n
        self.epsilon = eps
        self.ln_t = np.log(self.n)
        self.c = 1

    def epsilon_greedy(self, Q):
        rand = np.random.random()
        if rand < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(Q)
        return action

    def incremental_sample_rule(self, q_old, qta, step_size):
        return q_old + 1 / step_size * (qta - q_old)

    def soft_max(Q, tau):
        e_q_tau = np.exp(Q / tau)
        prob_actions = e_q_tau / np.sum(e_q_tau)

        cumulative_probability = 0.0
        choice = np.random.random()
        for a, pr in enumerate(prob_actions):
            cumulative_probability += pr
            if cumulative_probability > choice:
                return a

    def upper_confidence_bound(self, Q, Ntas):
        sq = []
        for nta in Ntas:
            if nta == 0:
                sq.append(0.)
            else:
                sq.append(np.sqrt(self.ln_t / nta))
        arg = Q + self.c * np.array(sq)
        return np.argmax(arg, axis=0)
