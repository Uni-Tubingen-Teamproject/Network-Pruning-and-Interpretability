import json
import torch
import torch.nn as nn

# Beispielmodell: GoogLeNet (Inception v1)
from torchvision.models import googlenet

# Beispiel JSON-Dateiname
json_filename = 'Pruning Rates/pruning_rates_local_structured_l1.json'
#json_filename = 'Pruning Rates/pruning_rates_local_unstructured_old.json'

# Laden der Pruning-Raten aus der JSON-Datei
with open(json_filename, 'r') as file:
    pruning_rates = json.load(file)

# Beispiel Datenstruktur, die die Anzahl der Gewichte für jedes Modul enthält
# Format: (Filteranzahl, Filterhöhe, Filterbreite, Eingangs-Kanäle)
weight_counts = {
    "conv1.conv": (64, 7, 7, 3),
    "conv2.conv": (64, 1, 1, 64),
    "conv3.conv": (192, 3, 3, 64),
    "inception3a.branch1.conv": (64, 1, 1, 192),
    "inception3a.branch2.0.conv": (96, 1, 1, 192),
    "inception3a.branch2.1.conv": (128, 3, 3, 96),
    "inception3a.branch3.0.conv": (16, 1, 1, 192),
    "inception3a.branch3.1.conv": (32, 3, 3, 16),
    "inception3a.branch4.1.conv": (32, 1, 1, 192),
    "inception3b.branch1.conv": (128, 1, 1, 256),
    "inception3b.branch2.0.conv": (128, 1, 1, 256),
    "inception3b.branch2.1.conv": (192, 3, 3, 128),
    "inception3b.branch3.0.conv": (32, 1, 1, 256),
    "inception3b.branch3.1.conv": (96, 3, 3, 32),
    "inception3b.branch4.1.conv": (64, 1, 1, 256),
    "inception4a.branch1.conv": (192, 1, 1, 480),
    "inception4a.branch2.0.conv": (96, 1, 1, 480),
    "inception4a.branch2.1.conv": (208, 3, 3, 96),
    "inception4a.branch3.0.conv": (16, 1, 1, 480),
    "inception4a.branch3.1.conv": (48, 3, 3, 16),
    "inception4a.branch4.1.conv": (64, 1, 1, 480),
    "inception4b.branch1.conv": (160, 1, 1, 512),
    "inception4b.branch2.0.conv": (112, 1, 1, 512),
    "inception4b.branch2.1.conv": (224, 3, 3, 112),
    "inception4b.branch3.0.conv": (24, 1, 1, 512),
    "inception4b.branch3.1.conv": (64, 3, 3, 24),
    "inception4b.branch4.1.conv": (64, 1, 1, 512),
    "inception4c.branch1.conv": (128, 1, 1, 512),
    "inception4c.branch2.0.conv": (128, 1, 1, 512),
    "inception4c.branch2.1.conv": (256, 3, 3, 128),
    "inception4c.branch3.0.conv": (24, 1, 1, 512),
    "inception4c.branch3.1.conv": (64, 3, 3, 24),
    "inception4c.branch4.1.conv": (64, 1, 1, 512),
    "inception4d.branch1.conv": (112, 1, 1, 512),
    "inception4d.branch2.0.conv": (144, 1, 1, 512),
    "inception4d.branch2.1.conv": (288, 3, 3, 144),
    "inception4d.branch3.0.conv": (32, 1, 1, 512),
    "inception4d.branch3.1.conv": (64, 3, 3, 32),
    "inception4d.branch4.1.conv": (64, 1, 1, 512),
    "inception4e.branch1.conv": (256, 1, 1, 528),
    "inception4e.branch2.0.conv": (160, 1, 1, 528),
    "inception4e.branch2.1.conv": (320, 3, 3, 160),
    "inception4e.branch3.0.conv": (32, 1, 1, 528),
    "inception4e.branch3.1.conv": (128, 3, 3, 32),
    "inception4e.branch4.1.conv": (128, 1, 1, 528),
    "inception5a.branch1.conv": (256, 1, 1, 832),
    "inception5a.branch2.0.conv": (160, 1, 1, 832),
    "inception5a.branch2.1.conv": (320, 3, 3, 160),
    "inception5a.branch3.0.conv": (32, 1, 1, 832),
    "inception5a.branch3.1.conv": (128, 3, 3, 32),
    "inception5a.branch4.1.conv": (128, 1, 1, 832),
    "inception5b.branch1.conv": (384, 1, 1, 832),
    "inception5b.branch2.0.conv": (192, 1, 1, 832),
    "inception5b.branch2.1.conv": (384, 3, 3, 192),
    "inception5b.branch3.0.conv": (48, 1, 1, 832),
    "inception5b.branch3.1.conv": (128, 3, 3, 48),
    "inception5b.branch4.1.conv": (128, 1, 1, 832),
}


# Berechnung der Gesamtzahl der Module
total_modules = len(pruning_rates)

# Berechnung der gewichteten Pruning-Rate
total_pruned_weights = 0
total_weights = 0

#factors = [0.4, 0.53, 0.66, 0.79, 0.92]
factors = [0.27, 0.55, 0.82, 1.1, 1.37, 1.65, 1.92]
for factor in factors:
    
    for module, prune_rate in pruning_rates.items():
        if module in weight_counts:
            filters, filter_height, filter_width, in_channels = weight_counts[module]
            num_weights = filters * filter_height * filter_width * in_channels
            total_pruned_weights += num_weights * prune_rate * factor
            total_weights += num_weights

    # Berechnung des durchschnittlichen Pruning-Anteils
    average_pruning_rate = total_pruned_weights / total_weights
    
    total_pruned_weights = 0
    total_weights = 0

    print(f"Factor: {factor}, Average Pruning Rate: {average_pruning_rate:.4f}")

