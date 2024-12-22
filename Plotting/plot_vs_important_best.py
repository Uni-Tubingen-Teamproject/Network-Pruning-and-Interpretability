import matplotlib.pyplot as plt

# Daten bleiben unverändert
pruning_rates = [0, 0.1889, 0.3795, 0.5711, 0.7617]

# Connection Sparsity SGD 50 epochs
conn_spars_sgd_50 = [0.7, 0.7238, 0.72276, 0.6921, 0.6359]

# Local Structured Adam 50 epochs
local_struct_adam_50 = [0.7, 0.67, 0.64, 0.57, 0.45]

# Local Structured SGD 50 epochs
local_struct_sgd_50 = [0.7, 0.7, 0.66, 0.59, 0.44]

# Global Unstructured Pruning (Last Epoch Accuracy)
global_unstructured_pruning = [0.7, 0.72488, 0.72226, 0.71658, 0.6757]

conn_spars_l1 = [0.7, 0.7240, 0.7218, 0.7229, 0.70506]

local_structured_not_retrained = [0.7, 0.003, 0.001, 0.001, 0.001]

connection_sparsity_not_retrained = [0.7, 0.0662, 0.0025, 0.0016, 0.0010]

# Vergrößere die Höhe der Grafik
plt.figure(figsize=(10, 7))

# Plot für Connection Sparsity
plt.plot(pruning_rates, conn_spars_sgd_50, marker='o', linestyle='--', color='limegreen',
         label='Connection Sparsity Random Adam')
plt.plot(pruning_rates, conn_spars_l1, marker='s', color='forestgreen',
         label='Connection Sparsity L1 SGD')

# Plot für Local Structured
plt.plot(pruning_rates, local_struct_adam_50, marker='^', linestyle='--', color='blue',
         label='Local Structured Adam')
plt.plot(pruning_rates, local_struct_sgd_50, marker='D', color='deepskyblue',
         label='Local Structured SGD')

# Plot für Unstructured
plt.plot(pruning_rates, global_unstructured_pruning, marker='x', color='red',
         label='Global Unstructured Pruning SGD')

# Plot für nicht retrainierte Daten
plt.plot(pruning_rates, local_structured_not_retrained, marker='v', linestyle='--', color='black',
         label='Local Structured Not Retrained')
plt.plot(pruning_rates, connection_sparsity_not_retrained, marker='*', color='gray',
         label='Connection Sparsity Not Retrained')

# Achsenbeschriftungen und Titel
plt.xlabel('Pruning Rate')
plt.ylabel('Accuracy')
plt.title('Pruning Rates vs. Accuracies')

# Position der Legende anpassen
plt.legend(loc='upper right', bbox_to_anchor=(
    0.37, -0.1), fancybox=True, shadow=True, ncol=1)
plt.ylim(0, 0.8)  # Y-Achse strecken
plt.grid(True)

# Speichern der Grafik als Vektorgrafik im PDF-Format
plt.savefig('pruning_rates_vs_accuracies.pdf',
            format='pdf', bbox_inches='tight')

plt.show()
