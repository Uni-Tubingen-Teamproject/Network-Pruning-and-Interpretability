import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Daten
pruning_rates = [0.1889, 0.3795, 0.5711, 0.7617]
accuracies_after_pruning = [0.0017, 0.0010, 0.0010, 0.0010]
accuracies_after_retraining = [
    [0.0017, 0.47892, 0.5169, 0.5462, 0.54914, 0.57276, 0.58436, 0.58338,
        0.60356, 0.61394, 0.6167],  # Pruning Rate 0.1889
    [0.0010, 0.44188, 0.50292, 0.52604, 0.55636, 0.5652, 0.57134,
        0.59178, 0.59694, 0.60868, 0.6150],  # Pruning Rate 0.3795
    [0.0010, 0.2428, 0.36156, 0.4035, 0.44596, 0.48422, 0.47854, 0.51472,
        0.52702, 0.536, 0.5572],    # Pruning Rate 0.5711
    [0.0010, 0.21694, 0.31338, 0.37824, 0.40556, 0.44728, 0.46504,
        0.49572, 0.50716, 0.5286, 0.5434]   # Pruning Rate 0.7617
]
epochs = np.arange(0, 11)

# Erstellen des 3D-Plots
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

for i, pruning_rate in enumerate(pruning_rates):
    ax.plot([pruning_rate]*11, epochs, accuracies_after_retraining[i],
            marker='o', label=f'Pruning Rate {pruning_rate:.4f}')

ax.set_title('Accuracy During Retraining vs Actual Pruning Rate and Epochs')
ax.set_xlabel('Actual Pruning Rate')
ax.set_ylabel('Epochs')
ax.set_zlabel('Accuracy')
ax.legend()
plt.show()
