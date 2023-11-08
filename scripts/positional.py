import numpy 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches


# draw sin(x) and cos(x) in the same figure, x range from 1 to 100

x = numpy.arange(0, 50, 1)
y = numpy.sin(x)
z = numpy.cos(x)

# add dot 
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y, color='blue')
ax.plot(x, z, color='red')
ax.scatter(x, y, color='blue')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('sin(x) and cos(x)')
plt.show()

