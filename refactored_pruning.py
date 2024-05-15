### Implementation GoogleNet and pruning techniques with the torch.nn.utils.prune package

# All required imports
import torch
import torchvision.datasets as datasets
import urllib
from PIL import Image
from torchvision import transforms
import json
import numpy as np
import torch.nn.utils.prune as prune


# Load the GoogleNet model
model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', weights = 'GoogLeNet_Weights.DEFAULT')
model.eval()

# Pre-process and normalise the input images
validation_transformation = transforms.Compose([
    transforms.Resize(256),               
    transforms.CenterCrop(224),          
    transforms.ToTensor(),                
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])  
])

# Define batch size, depending on available memory 
batch_size = 120

# Load the ImageNet validation set
validation_set = datasets.ImageNet(root = './data', split = 'val', transform = validation_transformation, download = True)

# Create a data loader for the validation set
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size = batch_size, shuffle = False, num_workers = 8)

# Load the ImageNet class labels
with open('imagenet_class_index.json') as f:
    class_idx = json.load(f)

classes = {idx: label for idx, label in class_idx.items()}


### Accuracy pre-pruning

# Define a function to compute the accuracy

def correct_pred(validation_loader, model): 

    correct_predictions = 0
    total_samples = 0

    for images, labels in validation_loader:
        if torch.cuda.is_available():
            images = images.to('cuda')
            labels = labels.to('cuda')

        with torch.no_grad():
            output = model(images)

        _, prediction = torch.max(output, 1)
        correct_predictions += (prediction == labels).sum().item()
        total_samples += labels.size(0)
    
    return correct_predictions, total_samples


# Compute accuracy pre-pruning
correct_predictions, total_samples = correct_pred(validation_loader, model)
accuracy = correct_predictions / total_samples

print(f"Accuracy: (accuracy:.2)")



### Pruning using torch.nn.utils.prune

# Percentage of parameters to be pruned
amounts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Determining the number of prunable modules 
module_count = sum(1 for _, module in model.named_modules() if hasattr(module, 'weight'))


## Unstructured Pruning

# Global-L1 Unstructured Pruning

# Array to store results, one-dimensional as we prune globally
results_global_unstructured_l1 = np.zeros(len(amounts))

parameters_to_prune = []
for module_name, module in model.named_modules():

    # Check, if module carries the attribute 'weight'
    if hasattr(module, 'weight') and module.weight is not None:
        parameters_to_prune.append((module, 'weight'))

# Loop through different pruning rates
for i, pruning_rate in enumerate(amounts):
    print(f"Pruning Rate: {pruning_rate}")
    
    # Prune the model
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method = prune.L1Unstructured,
        amount = pruning_rate,
    )

    # Count non-zero trainable parameters after pruning
    total_params = sum(p.numel() for p in model.parameters())
    non_zero_params = sum(p.numel() for p in model.parameters() if p.data.count_nonzero() > 0)
    pruned_params = total_params - non_zero_params

    print("Total parameters:", total_params)
    print("Non-zero parameters after pruning:", non_zero_params)
    print("Pruned parameters:", pruned_params)

    # Calculate the accuracy after pruning
    correct_predictions, total_samples = correct_pred(validation_loader, model)
    accuracy = correct_predictions / total_samples

    results_global_unstructured_l1[i] = accuracy

    # Reset the model to its original state (remove pruning)
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')

# Save the results
np.save('results_global_unstructured_l1.npy', results_global_unstructured_l1)


# Local-L1 Unstructured Pruning 

# Save the results in a vector 
results_local_unstructured_l1 = np.zeros(module_count, len(amounts))

# Loop through different pruning rates
for module_name, module in model.named_modules():

    # Given modules have weights, we prune them according to the amounts parameter and the l1-norm
    if hasattr(module, 'weight'):
        for i, pruning_rate in enumerate(amounts):
            prune.l1_unstructured(module, name = 'weight', amount = pruning_rate)

            # Assess the accuracy and store it 
            correct_predictions, total_samples = correct_pred(validation_loader, model)
            accuracy = correct_predictions / total_samples

            results_local_unstructured_l1[module_name][i] = accuracy

            # Reset the model to its original state (remove pruning)
            for module, _ in parameters_to_prune:
                prune.remove(module, 'weight')

# Save the results
np.save('results_local_unstructured_l1.npy', results_local_unstructured_l1)


# Local-Random Unstructured Pruning
# Number of runs per pruning amount to see some convergence in the accuracy value
runs = 2  

# Array for saving accuracy for each pruning amount and run
results_local_unstructured_random = np.zeros(module_count, len(amounts), runs)

# Prune the model
for module_name, module in model.named_modules():

    # Given modules have weights, we prune them randomly
    if hasattr(module, 'weight'):

        for i, pruning_rate in enumerate(amounts):

            # Simulate k runs
            for k in range(runs):
                # Apply pruning to the module
                prune.random_unstructured(module, name = 'weight', amount = pruning_rate)

                # Validate after pruning
                correct_predictions, total_samples = correct_pred(validation_loader, model)
                accuracy = correct_predictions / total_samples

                results_local_unstructured_random[module_name][i][k] = accuracy

                # Reset the model to its original state (remove pruning)
                for module, _ in parameters_to_prune:
                    prune.remove(module, 'weight')

                # Free GPU memory
                torch.cuda.empty_cache()

        # Output the average accuracies for each pruning amount
        mean_accuracy = np.mean(results_local_unstructured_random[module_name], axis = 1)
        std_deviation = np.std(results_local_unstructured_random[module_name], axis = 1)
        for i, amt in enumerate(amounts):
            print(f"Pruning amount: {amt}, Mean accuracy for module {module_name}: {mean_accuracy[i]}, Std deviation: {std_deviation[i]}")


# Save the pruning results
np.save('results_local_unstructured_random.npy', results_local_unstructured_random)



## Structured Pruning

# Local-L1 Structured Pruning
# Create an array to save the results, 2 = number of dims (0 = Convolutional Layers, 1 = Linear layers)
results_local_structured_l1 = np.zeros((module_count, len(amounts), 2))

for module_index, (module_name, module) in enumerate(model.named_modules()):

    if hasattr(module, 'weight'):
        for dim in [0, 1]:
            for i, pruning_rate in enumerate(amounts):
                prune.ln_structured(module, name = 'weight', amount = pruning_rate, n = 1, dim = dim)

                # Assess the accuracy and store it 
                correct_predictions, total_samples = correct_pred(validation_loader, model)
                accuracy = correct_predictions / total_samples

                results_local_structured_l1[module_index, i, dim] = accuracy

                # Reset the model to its original state (remove pruning)
                for module, _ in parameters_to_prune:
                    prune.remove(module, 'weight')

# save the results
np.save('results_local_structured_l1.npy', results_local_structured_l1)


# Local-L2 Structured Pruning
# Create an array to save the results, 2 = number of dims (0 = Convolutional Layers, 1 = Linear layers)
results_local_structured_l2 = np.zeros((module_count, len(amounts), 2))

for module_index, (module_name, module) in enumerate(model.named_modules()):

    if hasattr(module, 'weight'):
        for dim in [0, 1]:
            for i, pruning_rate in enumerate(amounts):
                prune.ln_structured(module, name = 'weight', amount = pruning_rate, n = 2, dim = dim)

                # Assess the accuracy and store it 
                correct_predictions, total_samples = correct_pred(validation_loader, model)
                accuracy = correct_predictions / total_samples

                results_local_structured_l2[module_index, i, dim] = accuracy

                # Reset the model to its original state (remove pruning)
                for module, _ in parameters_to_prune:
                    prune.remove(module, 'weight')

# save the results
np.save('results_local_structured_l2.npy', results_local_structured_l2)


# Local-Random Structured Pruning
# Number of runs per pruning amount to see some convergence in the accuracy value
runs = 2  

# Create an array to save the results
results_local_structured_random = np.zeros((module_count, len(amounts), 2, runs))

for module_index, (module_name, module) in enumerate(model.named_modules()):

    if hasattr(module, 'weight'):
        for dim in [0, 1]:
            for i, pruning_rate in enumerate(amounts):

                # Simulate k runs per pruning rate and module
                for k in range(runs):
                    prune.random_structured(module, name = 'weight', amount = pruning_rate, dim = dim)

                    # Assess the accuracy and store it 
                    correct_predictions, total_samples = correct_pred(validation_loader, model)
                    accuracy = correct_predictions / total_samples

                    results_local_structured_random[module_index, i, dim, k] = accuracy

                    # Reset the model to its original state (remove pruning)
                    for module, _ in parameters_to_prune:
                        prune.remove(module, 'weight')

# save the results
np.save('results_local_structured_random.npy', results_local_structured_random)