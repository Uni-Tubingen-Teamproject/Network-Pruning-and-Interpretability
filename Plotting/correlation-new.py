import pandas as pd

# Daten
pruning_rates = [0, 0.2, 0.4, 0.6, 0.8]

mis_confidence = {
    
    'Local Structured SGD 50 epochs': [0.699678, 0.673034, 0.684253, 0.705554, 0.749621],
    'Local Structured Adam 50 epochs': [0.699678, 0.703319, 0.692476, 0.709577, 0.739509],
    'Connection Sparsity random SGD 50 epochs': [0.699678, 0.670764, 0.670914, 0.678927, 0.687622],
    'Connection Sparsity L1 not retrained': [0.699678, 0.649680, 0.657086, 0.672725, 0.659024],
    'Connection Sparsity L1 50 epochs': [0.699678, 0.66876, 0.66879, 0.672433, 0.67687],
    'Global Unstructured 50 epochs': [0.699678, 0.67216, 0.670561, 0.68272, 0.697552],
    'Local Structured not retrained': [0.699678, 0.67145, 0.686136, 0.66566, 0.70404],
    'Local Structured iterative 4x50 epochs': [0.699678, 0.7, 0.7, 0.7, 0.714553]
}

# Leere Listen für Pruning Rates und MIS Confidence Scores erstellen
all_pruning_rates = []
all_mis_scores = []

# Daten in die Listen umwandeln
for method, scores in mis_confidence.items():
    for score in scores:
        all_pruning_rates.append(pruning_rates[scores.index(score)])
        all_mis_scores.append(score)

# DataFrame erstellen
df = pd.DataFrame({
    'Pruning Rate': all_pruning_rates,
    'MIS Confidence': all_mis_scores
})

# Korrelation berechnen
correlation = df['Pruning Rate'].corr(df['MIS Confidence'])
print(f'Korrelation zwischen Pruning Rate und MIS Confidence: {correlation}')
