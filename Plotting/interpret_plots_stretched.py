import matplotlib.pyplot as plt
import pandas as pd

# Daten
pruning_rates = [0.2, 0.4, 0.6, 0.8]

data_confidence = {
    'Unpruned GoogleNet': [0.699678, 0.699678, 0.699678, 0.699678],
    'Local Structured SGD 10 epochs': [0.677047, 0.685017, 0.702323, 0.730448],
    'Local Structured SGD 50 epochs': [0.673034, 0.684253, 0.705554, 0.749621],
    'Local Structured Adam 50 epochs': [0.703319, 0.692476, 0.709577, 0.739509],
    'Connection Sparsity SGD 50 epochs': [0.670764, 0.670914, 0.678927, 0.687622],
    'Connection Sparsity not retrained': [0.649680, 0.657086, 0.672725, 0.659024]
}

# Plot für MIS Confidence erstellen
plt.figure(figsize=(12, 8))

# Unpruned GoogleNet plotten
plt.axhline(y=data_confidence['Unpruned GoogleNet'][0],
            color='black', linestyle='--', label='Unpruned GoogleNet')

# Daten für verschiedene Methoden plotten
for method, conf_scores in data_confidence.items():
    if method != 'Unpruned GoogleNet':
        plt.plot(pruning_rates, conf_scores, marker='o', label=method)

plt.xlabel('Pruning Rate')
plt.ylabel('Avg MIS Confidence')
plt.title('Avg MIS Confidence vs Pruning Rate')
plt.ylim(0, 1)  # Skalierung der y-Achse von 0 bis 1
plt.legend()
plt.grid(True)

# Korrelation zwischen MIS Confidence und Pruning Rate berechnen
all_pruning_rates = []
all_conf_scores = []

for method, conf_scores in data_confidence.items():
    if method != 'Unpruned GoogleNet':
        all_pruning_rates.extend(pruning_rates)
        all_conf_scores.extend(conf_scores)

correlation_df_conf = pd.DataFrame({
    'Pruning Rate': all_pruning_rates,
    'MIS Confidence': all_conf_scores
})

correlation_conf = correlation_df_conf['Pruning Rate'].corr(
    correlation_df_conf['MIS Confidence'])
print(f'Correlation between Pruning Rate and MIS Confidence: {
      correlation_conf}')

# Plot anzeigen
plt.show()
