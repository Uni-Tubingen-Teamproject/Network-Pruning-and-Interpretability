import torch

# Pfad zum geprunten Modell
model_path = '/Users/jonathanvonrad/Desktop/Teamprojekt/Network Pruning and Interpretability/Network-Pruning-and-Interpretability/Pruned_Models/Local_Structured/Adam/10_Epochs/pruned_0.2_local_structured_Adam_retrained_10_epochs_model.pth'

# Modell laden
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

# Informationen über die geprunten Filter für jeden Layer
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
        layer_pruned_info.append(
            (name, original_filters, pruned_filters, remaining_filters, pruned_indices))

# Ausgabe der Informationen
for info in layer_pruned_info:
    layer_name, original_filters, pruned_filters, remaining_filters, pruned_indices = info
    print(f"Layer: {layer_name}")
    print(f"  Original Filters: {original_filters}")
    print(f"  Pruned Filters: {pruned_filters}")
    print(f"  Remaining Filters: {remaining_filters}")
    print(f"  Pruned Indices: {pruned_indices}")
    print()
