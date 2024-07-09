import matplotlib.pyplot as plt

# Daten für SGD 10 epochs
pruning_rates_sgd_10 = [0, 0.19134368949424185,
                        0.3435160210853824, 0.4652689953106288, 0.5629623619209634]
accuracies_sgd_10 = [0.7, 0.58758, 0.562, 0.51282, 0.49102]

# Daten für SGD 40 epochs
pruning_rates_sgd_40 = [0, 0.5728195543418653]
accuracies_sgd_40 = [0.7, 0.5836]

# Daten für iterative specific pr=0.2 epochs=10
pruning_rates_iterative = [0, 
    0.19623451332385755, 0.3407138700182427, 0.4503363140871395, 0.5351241172454702]
accuracies_iterative = [0.7, 0.5909, 0.5506, 0.5213, 0.4967]

# Diagramm erstellen
plt.figure(figsize=(12, 8))

# SGD 10 epochs
plt.plot(pruning_rates_sgd_10, accuracies_sgd_10, marker='o',
         linestyle='-', color='b', label='SGD 10 Epochs absolute')

# SGD 40 epochs
plt.plot(pruning_rates_sgd_40, accuracies_sgd_40, marker='o',
         linestyle='-', color='r', label='SGD 40 Epochs')

# Iterative specific pr=0.2 epochs=10
plt.plot(pruning_rates_iterative, accuracies_iterative, marker='o',
         linestyle='-', color='m', label='SGD 10 Epochs specific')

# Diagramm beschriften
plt.title('Accuracy after Retraining vs. Pruning Rate')
plt.xlabel('Pruning Rate')
plt.ylabel('Accuracy after Retraining')
plt.grid(True)
plt.legend()

# Diagramm anzeigen
plt.show()
