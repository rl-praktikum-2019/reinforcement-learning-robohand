import numpy as np

N = 0

def epsilon_greedy(eps, Q, env):
    rand = np.random.random()
    if rand < eps:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q)
    return action


# 1
def incremental_sample_rule(q_old, qta, step_size):
    return q_old + 1 / step_size * (qta - q_old)


# 1
def soft_max(Q, tau):
    e_q_tau = np.exp(Q / tau)
    prob_actions = e_q_tau / np.sum(e_q_tau)

    cumulative_probability = 0.0
    choice = np.random.random()
    for a, pr in enumerate(prob_actions):
        cumulative_probability += pr
        if cumulative_probability > choice:
            return a


# 2
C_param = 1
ln_t = np.log(N)


def upper_confidence_bound(Q, Ntas, c=None):
    if c is None:
        c = C_param

    sq = []
    for nta in Ntas:
        if nta == 0:
            sq.append(0.)
        else:
            sq.append(np.sqrt(ln_t / nta))
    arg = Q + c * np.array(sq)
    return np.argmax(arg, axis=0)