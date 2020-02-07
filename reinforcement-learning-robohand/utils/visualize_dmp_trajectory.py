import numpy as np
import matplotlib.pyplot as plt
import pydmps.dmp_discrete

y_des = np.array([[0.0, 0.0, 0.0], [0.0, -.15, .15]])
y_des -= y_des[:, 0][:, None]

dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=200, ay=np.ones(2) * 10.0)

dmp.imitate_path(y_des=y_des)

plt.figure(1, figsize=(6, 6))

y_track, dy_track, ddy_track = dmp.rollout(tau=5)
xs = np.arange(0, len(y_track), 1)

plt.plot(xs, y_track[:, 1] * 100, 'b--', lw=2, color='b')
plt.plot(xs, dy_track[:, 1]* 10, 'b--', lw=2, color='g')
plt.plot(xs, ddy_track[:, 1], 'b--', lw=2, color='r')


plt.title('DMP system - Wrist joint')
plt.xlabel('Step')
plt.ylabel('DMP result')
plt.legend(['y', 'dy', 'ddy'])
plt.show()
