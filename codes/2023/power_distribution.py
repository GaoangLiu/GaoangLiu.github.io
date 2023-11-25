import numpy as np
import matplotlib.pyplot as plt

# Set the parameters
alpha = 2
x_min = 1  # You can choose your own lower bound

# Generate random numbers following a power-law distribution
data_size = 1000
data = (np.random.pareto(alpha, data_size) + 1) * x_min

# Plot the histogram
plt.hist(data, bins=100, density=True, alpha=0.7, color='b')

# Plot the theoretical power-law distribution
x = np.linspace(x_min, np.max(data), 100)
y = x **(-alpha)
plt.plot(x, y, 'r-', linewidth=2)

# Add labels and title
plt.title(f'Power-law Distribution (α={alpha})')
plt.xlabel('Value')
plt.ylabel('Probability Density')

# Show the plot
plt.show()
