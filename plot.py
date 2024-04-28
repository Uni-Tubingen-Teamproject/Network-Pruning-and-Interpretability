import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Laden der Daten aus der results.npy-Datei
data = np.load('results.npy')

mean_array = np.mean(data, axis=-1)
fig = plt.figure()
ax = fig.add_subplot( projection='3d')
i = 0


# Verschachtelte For-Schleife
for x in np.arange(0, 1.0, 0.1):
    for y in np.arange(0, 1.0, 0.1):
        # Wandele x und y in ganzzahlige Indizes um
        x_index = int(x * 10)
        y_index = int(y * 10)
        ax.scatter(x, y, mean_array[x_index][y_index])
        # Greife auf das mean_array zu und verwende die ganzzahligen Indizes
	



ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
