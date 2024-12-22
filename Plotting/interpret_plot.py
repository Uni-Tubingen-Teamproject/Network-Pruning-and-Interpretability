import matplotlib.pyplot as plt
import pandas as pd
import itertools

# Daten
pruning_rates = [0, 0.2, 0.4, 0.6, 0.8]

mis_confidence = {
    'Unpruned GoogleNet': [0.699678, 0.699678, 0.699678, 0.699678],
    'Local Structured SGD 50 epochs': [0.699678, 0.673034, 0.684253, 0.705554, 0.749621],
    'Local Structured Adam 50 epochs': [0.699678, 0.703319, 0.692476, 0.709577, 0.739509],
    'Connection Sparsity random SGD 50 epochs': [0.699678, 0.670764, 0.670914, 0.678927, 0.687622],
    'Connection Sparsity L1 not retrained': [0.699678, 0.649680, 0.657086, 0.672725, 0.659024],
    'Connection Sparsity L1 50 epochs': [0.699678, 0.66876, 0.66879, 0.672433, 0.67687],
    'Global Unstructured 50 epochs': [0.699678, 0.67216, 0.670561, 0.68272, 0.697552],
    'Local Structured not retrained': [0.699678, 0.67145, 0.686136, 0.66566, 0.70404],
    'Local Structured iterative 4x50 epochs': [0.699678, 0.7, 0.7, 0.7, 0.714553]
}

# Farben und Symbole für die verschiedenen Kategorien
colors = {
    'Local Structured': itertools.cycle(['#1f77b4', '#aec7e8']),
    'Global Unstructured': itertools.cycle(['#d62728', '#ff9896']),
    'Connection Sparsity': itertools.cycle(['#2ca02c', '#98df8a']),
    'not retrained': '#000000'
}

markers = {
    'Local Structured SGD 50 epochs': 'o',
    'Local Structured Adam 50 epochs': 's',
    'Connection Sparsity random SGD 50 epochs': '^',
    'Connection Sparsity L1 not retrained': 'v',
    'Connection Sparsity L1 50 epochs': 'D',
    'Global Unstructured 50 epochs': 'x',
    'Local Structured not retrained': '*',
    'Local Structured iterative 4x50 epochs': 'p'
}

# Plot für MIS Confidence erstellen
plt.figure(figsize=(12, 8))

# Unpruned GoogleNet plotten
plt.axhline(y=mis_confidence['Unpruned GoogleNet'][0],
            color='black', linestyle='--', label='Unpruned GoogleNet')

# Daten für verschiedene Methoden plotten
for method, conf_scores in mis_confidence.items():
    if method != 'Unpruned GoogleNet':
        if 'not retrained' in method:
            color = colors['not retrained']
        elif 'Local Structured' in method:
            color = next(colors['Local Structured'])
        elif 'Global Unstructured' in method:
            color = next(colors['Global Unstructured'])
        elif 'Connection Sparsity' in method:
            color = next(colors['Connection Sparsity'])
        else:
            color = 'black'  # Default color if no category matches

        marker = markers.get(method, 'o')  # Verwende unterschiedliche Marker

        plt.plot(pruning_rates, conf_scores,
                 marker=marker, label=method, color=color)

plt.xlabel('Pruning Rate')
plt.ylabel('Avg MIS Confidence')
plt.title('Avg MIS Confidence vs Pruning Rate')
plt.legend()
plt.grid(True)

# Korrelation zwischen MIS Confidence und Pruning Rate berechnen
all_pruning_rates = []
all_conf_scores = []

for method, conf_scores in mis_confidence.items():
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

# Speichern der Grafik als Vektorgrafik im PDF-Format
plt.savefig('mis_confidence_vs_pruning_rate.pdf',
            format='pdf', bbox_inches='tight')

# Plot anzeigen
plt.show()
