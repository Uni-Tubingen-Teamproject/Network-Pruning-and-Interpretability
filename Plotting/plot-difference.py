import matplotlib.pyplot as plt

# Daten für das lokale unstrukturierte L1-Pruning
local_pruning_rate = [0.82, 0.72, 0.62, 0.52, 0.42]
local_accuracy = [0.00386, 0.14838, 0.48774, 0.62948, 0.68956]

# Daten für das globale unstrukturierte L1-Pruning
global_pruning_rate = [0.4, 0.5, 0.6, 0.7, 0.8]
global_accuracy = [0.66946, 0.6128, 0.4921, 0.10154, 0.0046]

# Plot erstellen
plt.figure(figsize=(10, 6))
plt.plot(local_pruning_rate, local_accuracy, marker='o',
         linestyle='-', label='Local Unstructured L1 Pruning Average')
plt.plot(global_pruning_rate, global_accuracy, marker='s',
         linestyle='--', label='Global Unstructured L1 Pruning')

# Titel und Achsenbeschriftungen hinzufügen
plt.title('Pruning Rate vs Accuracy')
plt.xlabel('Pruning Rate')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot anzeigen
plt.show()
