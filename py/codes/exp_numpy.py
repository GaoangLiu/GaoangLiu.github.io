#!/user/bin/python3
import numpy

arr = [1, 3, 4]
print("The sum: ", numpy.sum(arr))
print("The mean: ", numpy.mean(arr))
print("The size: ", numpy.size(arr))

# 9 numbers from 0 to 2
arr = numpy.linspace( 0, 2, 9)
print(arr)
print(len(arr))