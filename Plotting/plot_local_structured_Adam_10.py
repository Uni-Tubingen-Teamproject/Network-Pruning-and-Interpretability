import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Daten
pruning_rates = [0.19134368949424185, 0.3822005349688308, 0.5728195543418653, 0.7636763998164542]
accuracies_after_pruning = [0.00368, 0.001, 0.001, 0.001]
accuracies_after_retraining = [
    [0.00368, 0.22686, 0.261, 0.2925, 0.31978, 0.3321, 0.362, 0.38202, 0.40856, 0.41112, 0.46234],
    [0.001, 0.18822, 0.22876, 0.27064, 0.30788, 0.32964, 0.3479, 0.37052, 0.3967, 0.40934, 0.42572],
    [0.001, 0.14138, 0.1973, 0.22438, 0.259, 0.2822, 0.29624, 0.32348, 0.33802, 0.35222, 0.37184],
    [0.001, 0.09684, 0.143, 0.16406, 0.18358, 0.2111, 0.22302, 0.24204, 0.24762, 0.26884, 0.28936]
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
