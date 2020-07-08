import numpy as np 

# 概论论与统计 Chapter 6 
data = [0.5, 1.3, 0.6 ,1.7, 2.2, 1.2, 0.8, 1.5, 2.0, 1.6]
mean = np.mean(data)
var = np.var(data)
print('Mean and var of data:', mean, var)
theta = mean *  2
print('Theta estimated as', theta)
