import pickle
import pandas as pd
import torch

# Liste der Model-Pfade und Interpretierbarkeits-Scores-Pfade
model_paths = [
    '/mnt/qb/wichmann/tklein16/Pruned_Models/pruned_0.2_connection_sparsity_l1_SGD_retrained_50_epochs_model.pth',
    '/mnt/qb/wichmann/tklein16/Pruned_Models/pruned_0.2_connection_sparsity_model.pth',
    '/mnt/qb/wichmann/tklein16/Pruned_Models/pruned_0.2_global_unstructured_SGD_retrained_50_epochs_model.pth',
    '/mnt/qb/wichmann/tklein16/Pruned_Models/pruned_0.2_local_structured_model.pth',
    '/mnt/qb/wichmann/tklein16/Pruned_Models/pruned_0.33_local_structured_specific_SGD_retrained_iterative_4x50_epochs_model.pth',
    '/mnt/qb/wichmann/tklein16/Pruned_Models/pruned_0.4_connection_sparsity_l1_SGD_retrained_50_epochs_model.pth',
    '/mnt/qb/wichmann/tklein16/Pruned_Models/pruned_0.4_connection_sparsity_model.pth',
    '/mnt/qb/wichmann/tklein16/Pruned_Models/pruned_0.4_global_unstructured_SGD_retrained_50_epochs_model.pth',
    '/mnt/qb/wichmann/tklein16/Pruned_Models/pruned_0.4_local_structured_model.pth',
    '/mnt/qb/wichmann/tklein16/Pruned_Models/pruned_0.6_connection_sparsity_l1_SGD_retrained_50_epochs_model.pth',
    '/mnt/qb/wichmann/tklein16/Pruned_Models/pruned_0.6_connection_sparsity_model.pth',
    '/mnt/qb/wichmann/tklein16/Pruned_Models/pruned_0.6_global_unstructured_SGD_retrained_50_epochs_model.pth',
    '/mnt/qb/wichmann/tklein16/Pruned_Models/pruned_0.6_local_structured_model.pth',
    '/mnt/qb/wichmann/tklein16/Pruned_Models/pruned_0.8_connection_sparsity_l1_SGD_retrained_50_epochs_model.pth',
    '/mnt/qb/wichmann/tklein16/Pruned_Models/pruned_0.8_connection_sparsity_model.pth',
    '/mnt/qb/wichmann/tklein16/Pruned_Models/pruned_0.8_global_unstructured_SGD_retrained_50_epochs_model.pth',
    '/mnt/qb/wichmann/tklein16/Pruned_Models/pruned_0.8_local_structured_model.pth'
]

interpretability_paths = [
    '/mnt/qb/work/wichmann/wzz745/Network-Pruning-and-Interpretability/New_Interpret_Scores/machine_interpretability_dreamsim_natural_pruned_googlenet_Pruned_Models_pruned_0.2_connection_sparsity_l1_SGD_retrained_50_epochs_model.pkl',
    '/mnt/qb/work/wichmann/wzz745/Network-Pruning-and-Interpretability/New_Interpret_Scores/machine_interpretability_dreamsim_natural_pruned_googlenet_Pruned_Models_pruned_0.2_connection_sparsity_model.pkl',
    '/mnt/qb/work/wichmann/wzz745/Network-Pruning-and-Interpretability/New_Interpret_Scores/machine_interpretability_dreamsim_natural_pruned_googlenet_Pruned_Models_pruned_0.2_global_unstructured_SGD_retrained_50_epochs_model.pkl',
    '/mnt/qb/work/wichmann/wzz745/Network-Pruning-and-Interpretability/New_Interpret_Scores/machine_interpretability_dreamsim_natural_pruned_googlenet_Pruned_Models_pruned_0.2_local_structured_model.pkl',
    '/mnt/qb/work/wichmann/wzz745/Network-Pruning-and-Interpretability/New_Interpret_Scores/machine_interpretability_dreamsim_natural_pruned_googlenet_Pruned_Models_pruned_0.33_local_structured_specific_SGD_retrained_iterative_4x50_epochs_model.pkl',
    '/mnt/qb/work/wichmann/wzz745/Network-Pruning-and-Interpretability/New_Interpret_Scores/machine_interpretability_dreamsim_natural_pruned_googlenet_Pruned_Models_pruned_0.4_connection_sparsity_l1_SGD_retrained_50_epochs_model.pkl',
    '/mnt/qb/work/wichmann/wzz745/Network-Pruning-and-Interpretability/New_Interpret_Scores/machine_interpretability_dreamsim_natural_pruned_googlenet_Pruned_Models_pruned_0.4_connection_sparsity_model.pkl',
    '/mnt/qb/work/wichmann/wzz745/Network-Pruning-and-Interpretability/New_Interpret_Scores/machine_interpretability_dreamsim_natural_pruned_googlenet_Pruned_Models_pruned_0.4_global_unstructured_SGD_retrained_50_epochs_model.pkl',
    '/mnt/qb/work/wichmann/wzz745/Network-Pruning-and-Interpretability/New_Interpret_Scores/machine_interpretability_dreamsim_natural_pruned_googlenet_Pruned_Models_pruned_0.4_local_structured_model.pkl',
    '/mnt/qb/work/wichmann/wzz745/Network-Pruning-and-Interpretability/New_Interpret_Scores/machine_interpretability_dreamsim_natural_pruned_googlenet_Pruned_Models_pruned_0.6_connection_sparsity_l1_SGD_retrained_50_epochs_model.pkl',
    '/mnt/qb/work/wichmann/wzz745/Network-Pruning-and-Interpretability/New_Interpret_Scores/machine_interpretability_dreamsim_natural_pruned_googlenet_Pruned_Models_pruned_0.6_connection_sparsity_model.pkl',
    '/mnt/qb/work/wichmann/wzz745/Network-Pruning-and-Interpretability/New_Interpret_Scores/machine_interpretability_dreamsim_natural_pruned_googlenet_Pruned_Models_pruned_0.6_global_unstructured_SGD_retrained_50_epochs_model.pkl',
    '/mnt/qb/work/wichmann/wzz745/Network-Pruning-and-Interpretability/New_Interpret_Scores/machine_interpretability_dreamsim_natural_pruned_googlenet_Pruned_Models_pruned_0.6_local_structured_model.pkl',
    '/mnt/qb/work/wichmann/wzz745/Network-Pruning-and-Interpretability/New_Interpret_Scores/machine_interpretability_dreamsim_natural_pruned_googlenet_Pruned_Models_pruned_0.8_connection_sparsity_l1_SGD_retrained_50_epochs_model.pkl',
    '/mnt/qb/work/wichmann/wzz745/Network-Pruning-and-Interpretability/New_Interpret_Scores/machine_interpretability_dreamsim_natural_pruned_googlenet_Pruned_Models_pruned_0.8_connection_sparsity_model.pkl',
    '/mnt/qb/work/wichmann/wzz745/Network-Pruning-and-Interpretability/New_Interpret_Scores/machine_interpretability_dreamsim_natural_pruned_googlenet_Pruned_Models_pruned_0.8_global_unstructured_SGD_retrained_50_epochs_model.pkl',
    '/mnt/qb/work/wichmann/wzz745/Network-Pruning-and-Interpretability/New_Interpret_Scores/machine_interpretability_dreamsim_natural_pruned_googlenet_Pruned_Models_pruned_0.8_local_structured_model.pkl'
]

# Funktion, um das Modell zu laden und die pruned filters zu extrahieren


def load_model_and_extract_pruned_info(model_path):
    print(f"Lade Modell: {model_path}")
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()

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

    pruned_layers = {info[0]: info[4] for info in layer_pruned_info}
    return pruned_layers

# Funktion, um die Interpretierbarkeits-Scores zu berechnen


def calculate_interpretability_scores(file_path, pruned_layers):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    interpretability_scores_df = data['scores']

    def is_not_pruned(row):
        layer_name = row['layer']
        unit = int(row['unit'])
        if layer_name in pruned_layers:
            if unit in pruned_layers[layer_name]:
                return False
        return True

    filtered_df = interpretability_scores_df[interpretability_scores_df.apply(
        is_not_pruned, axis=1)]
    filtered_df = filtered_df[~filtered_df['layer'].str.contains('_bn')]

    numeric_df = filtered_df.select_dtypes(include='number')

    average_per_layer = numeric_df.groupby(filtered_df['layer']).mean()
    average_entire_network = numeric_df.mean()

    avg_mis_confidence = average_entire_network['mis_confidence']
    avg_mis_score = average_entire_network['mis_score']

    print(f"Durchschnittlicher Mis_confidence: {avg_mis_confidence}")
    print(f"Durchschnittlicher Mis_score: {avg_mis_score}")
    print("-" * 50)


for model_path, interpretability_path in zip(model_paths, interpretability_paths):
    pruned_layers = load_model_and_extract_pruned_info(model_path)
    calculate_interpretability_scores(interpretability_path, pruned_layers)
