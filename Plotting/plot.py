import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Laden der Daten aus der results.npy-Datei
data = np.load('results.npy')
data2 = np.load('results (1).npy')

# Print Data
# print("First run:", data, "\n")
# print("Second run:", data2)

mean_array = np.mean(data2, axis=-1)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Erstellung des X- und Y-Gitters für die Oberfläche
X, Y = np.meshgrid(np.arange(0, 1.0, 0.1), np.arange(0, 1.0, 0.1))

# Plot der Oberfläche
ax.plot_surface(X, Y, mean_array, cmap='viridis')

ax.set_xlabel('Conv 2')
ax.set_ylabel('Conv 1')
ax.set_zlabel('Accuracy')

plt.show()
