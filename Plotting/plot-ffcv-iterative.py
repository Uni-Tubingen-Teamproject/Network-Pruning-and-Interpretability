import matplotlib.pyplot as plt

# Daten für learning rate 0.001
data_lr_0_001 = [
    {"Actual Pruning Rate": 0.0, "Accuracy Before": 0.69772, "Accuracy After": 0.69772},
    {"Actual Pruning Rate": 0.383346122315363,
        "Accuracy Before": 0.66196, "Accuracy After": 0.6856},
    {"Actual Pruning Rate": 0.6114729409855514,
        "Accuracy Before": 0.34106, "Accuracy After": 0.67454},
    {"Actual Pruning Rate": 0.747780522881669,
        "Accuracy Before": 0.12764, "Accuracy After": 0.65176},
    {"Actual Pruning Rate": 0.8295453114682544,
        "Accuracy Before": 0.01864, "Accuracy After": 0.61948},
    {"Actual Pruning Rate": 0.8787793365491153,
        "Accuracy Before": 0.00454, "Accuracy After": 0.58192}
]

# Daten für learning rate 0.01
data_lr_0_01 = [
    {"Actual Pruning Rate": 0.0, "Accuracy Before": 0.69772, "Accuracy After": 0.69772},
    {"Actual Pruning Rate": 0.383346122315363,
        "Accuracy Before": 0.66196, "Accuracy After": 0.65128},
    {"Actual Pruning Rate": 0.6114729409855514,
        "Accuracy Before": 0.55224, "Accuracy After": 0.64718},
    {"Actual Pruning Rate": 0.747780522881669,
        "Accuracy Before": 0.379, "Accuracy After": 0.6278},
    {"Actual Pruning Rate": 0.8295453114682544,
        "Accuracy Before": 0.13406, "Accuracy After": 0.6083},
    {"Actual Pruning Rate": 0.8787793365491153,
        "Accuracy Before": 0.03928, "Accuracy After": 0.58216}
]

# Extrahiere die Daten


def extract_data(data):
    pruning_rates = [item["Actual Pruning Rate"] for item in data]
    accuracies_before = [item["Accuracy Before"] for item in data]
    accuracies_after = [item["Accuracy After"] for item in data]
    return pruning_rates, accuracies_before, accuracies_after


pruning_rates_0_001, accuracies_before_0_001, accuracies_after_0_001 = extract_data(
    data_lr_0_001)
pruning_rates_0_01, accuracies_before_0_01, accuracies_after_0_01 = extract_data(
    data_lr_0_01)

# Erstelle das Diagramm
plt.figure(figsize=(12, 8))

# Plot für learning rate 0.001
for i in range(len(pruning_rates_0_001)):
    plt.plot([pruning_rates_0_001[i], pruning_rates_0_001[i]], [accuracies_before_0_001[i], accuracies_after_0_001[i]],
             marker='o', linestyle='-', color='blue', label='Learning Rate 0.001' if i == 0 else "")

for i in range(len(pruning_rates_0_001) - 1):
    plt.plot([pruning_rates_0_001[i], pruning_rates_0_001[i + 1]], [accuracies_after_0_001[i], accuracies_before_0_001[i + 1]],
             marker='o', linestyle='--', color='blue')

# Plot für learning rate 0.01
for i in range(len(pruning_rates_0_01)):
    plt.plot([pruning_rates_0_01[i], pruning_rates_0_01[i]], [accuracies_before_0_01[i], accuracies_after_0_01[i]],
             marker='s', linestyle='-', color='red', label='Learning Rate 0.01' if i == 0 else "")

for i in range(len(pruning_rates_0_01) - 1):
    plt.plot([pruning_rates_0_01[i], pruning_rates_0_01[i + 1]], [accuracies_after_0_01[i], accuracies_before_0_01[i + 1]],
             marker='s', linestyle='--', color='red')

plt.title('Accuracy vs. Actual Pruning Rate Before and After Retraining (Learning Rate 0.001 vs 0.01)')
plt.xlabel('Actual Pruning Rate')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
