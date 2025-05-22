import matplotlib.pyplot as plt

# Daten für die Achsen
pruning_rate = [0.0, 0.442, 0.442, 0.6024, 0.6024, 0.74754,
                0.74754, 0.83812, 0.83812, 0.8906, 0.8906]
accuracy = [0.7, 0.166, 0.71938, 0.69024,
            0.718, 0.53044, 0.7082, 0.18154, 0.6859, 0.063, 0.67]

# Plot erstellen
plt.figure(figsize=(8, 6))
plt.plot(pruning_rate, accuracy, marker='o', linestyle='-', color='b')
plt.title('Accuracy vs. Actual Pruning Rate')
plt.xlabel('Actual Pruning Rate')
plt.ylabel('Accuracy')
plt.grid(True)

# Speichern der Grafik als Vektorgrafik im PDF-Format
plt.savefig('accuracy_vs_pruning_rate.pdf', format='pdf')

plt.show()
