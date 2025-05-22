import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Daten
pruning_rates = [0.19134368949424185, 0.3822005349688308, 0.5728195543418653, 0.7636763998164542]
accuracies_after_pruning = [0.00368, 0.001, 0.001, 0.001]
accuracies_after_retraining = [
    [0.00368, 0.43072, 0.47454, 0.48712, 0.514, 0.5399, 0.54416, 0.55292, 0.56772, 0.58572, 0.5932],
    [0.001, 0.3324, 0.40424, 0.4239, 0.44198, 0.47322, 0.47908, 0.506, 0.51368, 0.5366, 0.53488],
    [0.001, 0.17806, 0.26242, 0.31136, 0.36236, 0.36594, 0.38386, 0.40608, 0.41974, 0.43736, 0.45186],
    [0.001, 0.0755, 0.12706, 0.16438, 0.173, 0.205, 0.23544, 0.25908, 0.2696, 0.26922, 0.29024]
]

epochs = np.arange(0, 11)

# Erstellen des 3D-Plots
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

for i, pruning_rate in enumerate(pruning_rates):
    ax.plot([pruning_rate]*11, epochs, accuracies_after_retraining[i], marker='o', label=f'Pruning Rate {pruning_rate:.4f}')

ax.set_title('Accuracy During Retraining vs Actual Pruning Rate and Epochs')
ax.set_xlabel('Actual Pruning Rate')
ax.set_ylabel('Epochs')
ax.set_zlabel('Accuracy')
ax.legend()
plt.show()
