#!/user/bin/env python
import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot([1,2,3,4])
plt.ylabel('some numbers')
# plt.show()
fig.savefig("media/sample-a.png")