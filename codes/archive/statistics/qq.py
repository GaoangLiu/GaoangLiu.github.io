from numpy.random import seed
from numpy.random import randn
from matplotlib import pyplot
import scipy.stats as stats 
import pylab
import random 

# seed the random number generator
seed(1)
# generate univariate observations
data = 5 * randn(100) + 50
# print(data)
# histogram plot
# pyplot.hist(data)
# pyplot.show()

data = [random.randint(0, 100) for _ in range(100_000)]

subdata = random.sample(data, 1000)


stats.probplot(data, dist="norm", plot=pylab)
pyplot.show()
