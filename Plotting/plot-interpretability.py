import matplotlib.pyplot as plt

# Daten für SGD 50 Epochen
pruning_rates = [0, 0.19134368949424185, 0.3822005349688308,
                 0.5728195543418653, 0.7636763998164542]
interpretability_scores_sgd = [0.7407771084337349, 0.715, 0.725, 0.749, 0.792]
mis_confidence_sgd = [0.699, 0.672, 0.683, 0.704, 0.748]
mis_complete_sgd = [0.53, 0.499, 0.514, 0.545, 0.61]

# Daten für Adam 10 Epochen
interpretability_scores_adam = [0.740, 0.726, 0.732, 0.758, 0.781]
mis_confidence_adam = [0.699, 0.685, 0.689, 0.713, 0.734]
mis_complete_adam = [0.53, 0.514, 0.521, 0.557, 0.589]

# Erstellen der Visualisierung
fig, ax = plt.subplots(figsize=(12, 8))

# Plot für SGD
ax.plot(pruning_rates, interpretability_scores_sgd, marker='o',
        label='SGD - MIS Avg', color='b')
ax.plot(pruning_rates, mis_confidence_sgd, marker='s',
        label='SGD - MIS Confidence', color='r')
ax.plot(pruning_rates, mis_complete_sgd, marker='^',
        label='SGD - MIS Complete', color='g')

# Plot für Adam
ax.plot(pruning_rates, interpretability_scores_adam, marker='D',
        label='Adam - MIS Avg', color='b', linestyle='--')
ax.plot(pruning_rates, mis_confidence_adam, marker='X',
        label='Adam - MIS Confidence', color='r', linestyle='--')
ax.plot(pruning_rates, mis_complete_adam, marker='*',
        label='Adam - MIS Complete', color='g', linestyle='--')

ax.set_title('Comparison of MIS Avg, MIS Confidence, and MIS Complete vs Pruning Rates for SGD (50 Epochs) and Adam (10 Epochs)')
ax.set_xlabel('Pruning Rates')
ax.set_ylabel('Scores')
ax.legend()
ax.grid(True)

plt.show()
