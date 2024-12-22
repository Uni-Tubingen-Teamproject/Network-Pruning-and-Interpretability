import matplotlib.pyplot as plt

# Daten für Epochen und Validation Accuracy
epochs = list(range(1, 61))
validation_accuracy = [
    0.4493, 0.48216, 0.51376, 0.50666, 0.52704, 0.54264, 0.55242, 0.55932,
    0.56516, 0.57436, 0.58906, 0.5873, 0.59694, 0.60288, 0.60224, 0.61566,
    0.6164, 0.62184, 0.62438, 0.62348, 0.63302, 0.63656, 0.63552, 0.64102,
    0.64452, 0.64358, 0.64754, 0.64642, 0.6499, 0.65092, 0.65168, 0.65278,
    0.65512, 0.6521, 0.65736, 0.65708, 0.65676, 0.65726, 0.65882, 0.66024,
    0.65856, 0.66158, 0.6616, 0.66198, 0.663, 0.66152, 0.65982, 0.66346,
    0.6623, 0.66364, 0.66104, 0.66392, 0.66092, 0.6623, 0.66334, 0.66392,
    0.66386, 0.66394, 0.66314, 0.66324
]

# Plot-Erstellung
plt.figure(figsize=(10, 6))
plt.plot(epochs, validation_accuracy, marker='o',
         linestyle='-', color='b', label='Validation Accuracy')
plt.title('Validation Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.xticks(range(0, 61, 5))  # Setzt x-Achsen-Ticks alle 5 Epochen
plt.legend()

# Speichern der Grafik als Vektorgrafik im PDF-Format
plt.savefig('validation_accuracy_over_epochs.pdf', format='pdf')

plt.show()
