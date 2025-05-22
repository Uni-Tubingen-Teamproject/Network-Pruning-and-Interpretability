

# pruning_rates = [0.1889, 0.3795, 0.5711, 0.7617]

# # Connection Sparsity SGD 10 epochs  lr 0.01
# last_elements = [0.6167, 0.615, 0.5572, 0.5434]

# # Connection Sparsity SGD 50 epochs lr 0.01
# last_elements_2 = [0.7238, 0.72276, 0.6921, 0.6359]

# # Local Structured Adam 10 epochs lr 0.001
# last_elements_3 = [0.46234, 0.42572, 0.37184, 0.28936]

# # Local Structured Adam 50 epochs lr 0.001
# last_elements_4 = [0.67, 0.64, 0.57, 0.45]

# # Local Structured SGD 10 epochs lr 0.01
# last_elements_5 = [0.59, 0.53, 0.45, 0.29]

# # Local Structured SGD 50 epochs lr 0.01
# last_elements_6 = [0.7, 0.66, 0.59, 0.44]

# # Local Structured SGD 50 epochs lr 0.001
# last_elements_7 = [0.69, 0.637, 0.535, 0.325]

import matplotlib.pyplot as plt

pruning_rates = [0, 0.1889, 0.3795, 0.5711, 0.7617]

# Daten
last_elements_1 = [0.7, 0.6167, 0.615, 0.5572, 0.5434]
last_elements_2 = [0.7, 0.7238, 0.72276, 0.6921, 0.6359]
last_elements_3 = [0.7, 0.46234, 0.42572, 0.37184, 0.28936]
last_elements_4 = [0.7, 0.67, 0.64, 0.57, 0.45]
last_elements_5 = [0.7, 0.59, 0.53, 0.45, 0.29]
last_elements_6 = [0.7, 0.7, 0.66, 0.59, 0.44]
last_elements_7 = [0.7, 0.69, 0.637, 0.535, 0.325]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(pruning_rates, last_elements_1, marker='o',
         label='Connection Sparsity SGD 10 epochs lr 0.01')
plt.plot(pruning_rates, last_elements_2, marker='o',
         label='Connection Sparsity SGD 50 epochs lr 0.01')
plt.plot(pruning_rates, last_elements_3, marker='o',
         label='Local Structured Adam 10 epochs lr 0.001')
plt.plot(pruning_rates, last_elements_4, marker='o',
         label='Local Structured Adam 50 epochs lr 0.001')
plt.plot(pruning_rates, last_elements_5, marker='o',
         label='Local Structured SGD 10 epochs lr 0.01')
plt.plot(pruning_rates, last_elements_6, marker='o',
         label='Local Structured SGD 50 epochs lr 0.01')
plt.plot(pruning_rates, last_elements_7, marker='o',
         label='Local Structured SGD 50 epochs lr 0.001')

# Achsenbeschriftungen und Titel
plt.xlabel('Pruning Rates')
plt.ylabel('Accuracies (Last Elements)')
plt.title('Pruning Rates vs. Accuracies')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()




