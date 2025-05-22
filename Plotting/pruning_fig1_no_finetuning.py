import matplotlib.pyplot as plt

# Daten für den Graphen
pruning_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
accuracy = [0.7, 0.698, 0.69, 0.66, 0.615, 0.49, 0.1, 0.01, 0.003]

# Plot erstellen
plt.figure()
# 'bo-' für blaue Linie mit Kreisen
plt.plot(pruning_rate, accuracy, 'bo-', linewidth=1.5)
plt.title('Pruning Rate vs Accuracy')
plt.xlabel('Pruning Rate')
plt.ylabel('Accuracy')
plt.grid(True)

# Speichern als Vektorgrafik im PDF-Format
plt.savefig('pruning_vs_accuracy.pdf', format='pdf')

plt.show()
