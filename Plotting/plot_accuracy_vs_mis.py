import matplotlib.pyplot as plt
import pandas as pd

# Daten für MIS Confidence
pruning_rates_conf = [0.2, 0.4, 0.6, 0.8]

data_confidence = {
    'Unpruned GoogleNet': [0.699678, 0.699678, 0.699678, 0.699678],
    'Local Structured SGD 50 epochs': [0.673034, 0.684253, 0.705554, 0.749621],
    'Local Structured Adam 50 epochs': [0.703319, 0.692476, 0.709577, 0.739509],
    'Connection Sparsity SGD 50 epochs': [0.670764, 0.670914, 0.678927, 0.687622],
    'Connection Sparsity not retrained': [0.649680, 0.657086, 0.672725, 0.659024]
}

# Daten für Accuracy
pruning_rates_acc = [0.2, 0.4, 0.6, 0.8]

data_accuracy = {
    'Connection Sparsity SGD 50 epochs': [0.7238, 0.72276, 0.6921, 0.6359],
    'Local Structured Adam 50 epochs': [0.67, 0.64, 0.57, 0.45],
    'Local Structured SGD 50 epochs': [0.7, 0.66, 0.59, 0.44],
    'Connection Sparsity L1': [0.7240, 0.7218, 0.7229, 0.70506],
    'Connection Sparsity not retrained': [0.0037, 0.001, 0.001, 0.001]
}

# Plot für MIS Confidence erstellen
plt.figure(figsize=(12, 8))

# Unpruned GoogleNet plotten
plt.axhline(y=data_confidence['Unpruned GoogleNet'][0], color='black',
            linestyle='--', label='Unpruned GoogleNet (MIS Confidence)')

# Daten für verschiedene Methoden plotten (MIS Confidence)
for method, conf_scores in data_confidence.items():
    if method != 'Unpruned GoogleNet':
        plt.plot(pruning_rates_conf, conf_scores, marker='o', label=method)

# Daten für verschiedene Methoden plotten (Accuracy)
for method, acc_scores in data_accuracy.items():
    plt.plot(pruning_rates_acc, acc_scores, marker='x',
             linestyle='--', label=f"{method} (Accuracy)")

plt.xlabel('Pruning Rate')
plt.ylabel('Value')
plt.title('Avg MIS Confidence and Accuracy vs Pruning Rate')
plt.ylim(0, 1)
plt.legend()
plt.grid(True)

# Korrelation zwischen MIS Confidence und Pruning Rate berechnen
all_pruning_rates_conf = []
all_conf_scores = []

for method, conf_scores in data_confidence.items():
    if method != 'Unpruned GoogleNet':
        all_pruning_rates_conf.extend(pruning_rates_conf)
        all_conf_scores.extend(conf_scores)

correlation_df_conf = pd.DataFrame({
    'Pruning Rate': all_pruning_rates_conf,
    'MIS Confidence': all_conf_scores
})

correlation_conf = correlation_df_conf['Pruning Rate'].corr(
    correlation_df_conf['MIS Confidence'])
print(f'Correlation between Pruning Rate and MIS Confidence: {
      correlation_conf}')

# Plot anzeigen
plt.show()
