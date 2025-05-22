import matplotlib.pyplot as plt

# Daten
pruning_rates = [0, 0.2, 0.4, 0.6, 0.8]

accuracy_scores = {
    'Connection Sparsity SGD 50 epochs': [0.7, 0.7238, 0.72276, 0.6921, 0.6359],
    'Local Structured Adam 50 epochs': [0.7, 0.67, 0.64, 0.57, 0.45],
    'Local Structured SGD 50 epochs': [0.7, 0.7, 0.66, 0.59, 0.44],
    'Global Unstructured Pruning (Last Epoch Accuracy)': [0.7, 0.72488, 0.72226, 0.71658, 0.6757],
    'Connection Sparsity L1': [0.7, 0.7240, 0.7218, 0.7229, 0.70506],
    'Local Structured Not Retrained': [0.7, 0.003, 0.001, 0.001, 0.001],
    'Connection Sparsity Not Retrained': [0.7, 0.0662, 0.0025, 0.0016, 0.0010],
}

mis_scores = {
    'Unpruned GoogleNet': [0.699678, 0.699678, 0.699678, 0.699678],
    'Local Structured SGD 50 epochs': [0.699678, 0.673034, 0.684253, 0.705554, 0.749621],
    'Local Structured Adam 50 epochs': [0.699678, 0.703319, 0.692476, 0.709577, 0.739509],
    'Connection Sparsity random SGD 50 epochs': [0.699678, 0.670764, 0.670914, 0.678927, 0.687622],
    'Connection Sparsity L1 not retrained': [0.699678, 0.649680, 0.657086, 0.672725, 0.659024],
    'Connection Sparsity L1 50 epochs': [0.699678, 0.66876, 0.66879, 0.672433, 0.67687],
    'Global Unstructured 50 epochs': [0.699678, 0.67216, 0.670561, 0.68272, 0.697552],
    'Local Structured not retrained': [0.699678, 0.67145, 0.686136, 0.66566, 0.70404],
    'Local Structured iterative 4x50 epochs': [0.699678, 0.7, 0.7, 0.7, 0.714553],
}

# Plotting
plt.figure(figsize=(12, 8))

for method, accuracies in accuracy_scores.items():
    if method in mis_scores:
        plt.plot(accuracies, mis_scores[method],
                 marker='o', linestyle='-', label=method)

plt.xlabel('Accuracy')
plt.ylabel('MIS Score')
plt.title('Accuracy vs MIS Score for Different Pruning Methods')
plt.legend()
plt.grid(True)
plt.show()
