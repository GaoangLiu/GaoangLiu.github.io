import numpy as np
import matplotlib.pyplot as plt
ax = plt.gca()
# Keep the bottom and left spines, remove the top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
np.random.seed(0)
x1 = np.random.normal(-10, 1, 100)
x2 = np.random.normal(10, 1, 100)
x = np.concatenate((x1, x2))

y = np.random.uniform(1, 2, 200)
zeros = np.zeros_like(x)

plt.scatter(x, y, marker='.', color='b')

q = plt.quiver(zeros, zeros, x, y, angles='xy', scale_units='xy', scale=1, headlength=0, headaxislength=0)
key = plt.quiverkey(q, X=0.9, Y=0.9, U=1, label=None, labelpos='E')
plt.title("2D Vector Plot")
plt.xlabel("x")
plt.ylabel("y")

plt.xlim(-25, 25)
plt.ylim(0, 3)

# plt.show()
plt.savefig('/tmp/x.png', dpi=300)
