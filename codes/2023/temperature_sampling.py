import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 3, 5, 10, 3, 2, 4])
p_x = x / np.sum(x)
print(p_x)
taus = [0.1, 0.5, 0.9]
bar_width = 0.5  # Adjust the width of the bars
plt.figure(figsize=(12, 8))
# Plot x
plt.subplot(2, 2, 1)
plt.bar(np.arange(len(x)), x, width=bar_width, color='blue', alpha=0.7)
plt.title('$x$')
# Plot y for different values of tau
color_map = plt.get_cmap('hot')
for i, tau in enumerate(taus):
    p_y = np.exp(p_x / tau)
    y_sum = np.sum(p_y)
    p_y = p_y / y_sum
    print(p_y)
    plt.subplot(2, 2, i + 2)
    colors = color_map(p_y)  # using hot colormap for dynamic colors
    plt.bar(np.arange(len(p_y)), p_y, width=bar_width, color=colors, alpha=0.7)
    plt.title(f'$y(\\tau={tau}$)')  # Using LaTeX notation for tau symbol

# plt.tight_layout()
# plt.show()
plt.savefig('temperature_sampling.png', dpi=300)
