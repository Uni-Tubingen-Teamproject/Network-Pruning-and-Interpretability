import matplotlib.pyplot as plt

pruning_rates = [0, 0.1889, 0.3795, 0.5711, 0.7617]

# Connection Sparsity SGD 50 epochs
conn_spars_sgd_50_001 = [0.7, 0.7238, 0.72276, 0.6921, 0.6359]

# Local Structured Adam 50 epochs
local_struct_adam_50_0001 = [0.7, 0.67, 0.64, 0.57, 0.45]

# Local Structured SGD 50 epochs
local_struct_sgd_50_001 = [0.7, 0.7, 0.66, 0.59, 0.44]

# Global Unstructured Pruning (Last Epoch Accuracy)
global_unstructured_pruning = [0.7, 0.72488, 0.72226, 0.71658, 0.6757]


# Connection Sparsity L1
conn_spars_l1 = [0.7, 0.7240, 0.7218, 0.7229, 0.70506]

plt.figure(figsize=(10, 6))

# Plot für Connection Sparsity
plt.plot(pruning_rates, conn_spars_sgd_50_001, marker='o', linestyle='--', color='limegreen',
         label='Connection Sparsity Random Adam')
plt.plot(pruning_rates, conn_spars_l1, marker='o', color='forestgreen',
         label='Connection Sparsity L1 SGD')

# Plot für Local Structured
plt.plot(pruning_rates, local_struct_adam_50_0001, marker='o', linestyle='--', color='blue',
         label='Local Structured Adam')
plt.plot(pruning_rates, local_struct_sgd_50_001, marker='o', color='deepskyblue',
         label='Local Structured SGD')

# Plot für Unstructured
plt.plot(pruning_rates, global_unstructured_pruning, marker='o', color='red',
         label='Global Unstructured Pruning SGD')


# Achsenbeschriftungen und Titel
plt.xlabel('Pruning Rates')
plt.ylabel('Accuracies (Last Elements)')
plt.title('Pruning Rates vs. Accuracies')
plt.ylim(0, 1)  # Skalierung der y-Achse von 0 bis 1
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()
