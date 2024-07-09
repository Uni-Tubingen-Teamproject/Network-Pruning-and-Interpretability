import pickle
import pandas as pd
import torch

# Path to the pruned model
model_path = '/Users/jonathanvonrad/Desktop/Teamprojekt/Network Pruning and Interpretability/Network-Pruning-and-Interpretability/Pruned_Models/Local_Structured/Adam/10_Epochs/pruned_0.2_local_structured_Adam_retrained_10_epochs_model.pth'

# Load the model
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

# Information about the pruned filters for each layer
layer_pruned_info = []

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        original_filters = module.weight.size(0)
        remaining_filters_mask = (module.weight.abs().sum(
            dim=(1, 2, 3)) > 0).cpu().numpy()
        remaining_filters = remaining_filters_mask.sum()
        pruned_filters = original_filters - remaining_filters
        pruned_indices = [i for i, x in enumerate(
            remaining_filters_mask) if not x]
        layer_pruned_info.append((name.replace(
            '.', '_'), original_filters, pruned_filters, remaining_filters, pruned_indices))

# Mapping of layer names and pruned indices
pruned_layers = {info[0]: info[4] for info in layer_pruned_info}

# Path to the interpretability file
file_path = '/Users/jonathanvonrad/Desktop/Teamprojekt/Network Pruning and Interpretability/Network-Pruning-and-Interpretability/Interpretability Scores/Local Structured/Adam/10_Epochs/machine_interpretability_dreamsim_natural_pruned_googlenet_10_Epochs_pruned_0.2_local_structured_Adam_retrained_10_epochs_model.pkl'

# Load the file
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Extract DataFrame
interpretability_scores_df = data['scores']

# Remove pruned filters


def is_not_pruned(row):
    layer_name = row['layer']
    # Ensure 'unit' is treated as an integer
    unit = int(row['unit'])
    if layer_name in pruned_layers:
        if unit in pruned_layers[layer_name]:
            return False
    return True


# Filter non-pruned filters
filtered_df = interpretability_scores_df[interpretability_scores_df.apply(
    is_not_pruned, axis=1)]

# Filter out BatchNorm layers
filtered_df = filtered_df[~filtered_df['layer'].str.contains('_bn')]

# Filter numeric columns
numeric_df = filtered_df.select_dtypes(include='number')

# Calculate average values per layer
average_per_layer = numeric_df.groupby(filtered_df['layer']).mean()

# Calculate average values for the entire network
average_entire_network = numeric_df.mean()

# Print results
print("Average values per layer (excluding BatchNorm layers and pruned filters):")
print(average_per_layer)

print("\nAverage values for the entire network (excluding BatchNorm layers and pruned filters):")
print(average_entire_network)
