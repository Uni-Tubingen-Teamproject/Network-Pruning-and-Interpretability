import time
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torch.nn.utils.prune as prune
import numpy as np
import json
import os
from torchvision.datasets import ImageNet
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.imagenet import load_meta_file
import copy

class HackyImageNet(ImageNet):

    def __init__(self, root: str, devkit_loc="/mnt/qb/datasets/ImageNet2012/", split: str = 'train', transform=None, download=False, **kwargs):
        if download:
            raise NotImplementedError("Automatic download of the ImageNet dataset is not supported.")
        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train", "val"))
        self.devkit_loc = devkit_loc

        wnid_to_classes = load_meta_file(self.devkit_loc)[0]

        super(ImageNet, self).__init__(self.split_folder, **kwargs)
        self.transform = transform
        self.root = root
        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}

def setUpNeuralNetwork():   

    # Load the GoogleNet model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet',
                           weights='GoogLeNet_Weights.DEFAULT')
    model.eval()

    # Check if CUDA is available and move the model to GPU if it is
    if torch.cuda.is_available():
        model = model.to('cuda')

    # Pre-process and normalise the input images
    validation_transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # Define batch size, depending on available memory
    batch_size = 120

    # Beispiel zur Verwendung:
    root_path = "/scratch_local/datasets/ImageNet2012"

    validation_set = HackyImageNet(
        root=root_path, split='val', transform=validation_transformation)

    # Create a data loader for the validation set
    validation_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=batch_size, shuffle=False, num_workers=8)

    # Load the ImageNet class labels
    with open('imagenet_class_index.json') as f:
        class_idx = json.load(f)

    classes = {idx: label for idx, label in class_idx.items()}

    # Get the number of images in the validation set
    validation_set_size = len(validation_set)
    print("Size of the ImageNet validation set:", validation_set_size)

    # Compute accuracy pre-pruning
    correct_predictions, total_samples = correctPred(validation_loader, model)
    accuracy = correct_predictions / total_samples

    print(f"Accuracy before Pruning: {accuracy}")

    return model, validation_loader, validation_set, classes

def correctPred(validation_loader, model): 

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

model, validation_loader, validation_set, classes = setUpNeuralNetwork()

def setUpPruning(model):
    # Percentage of parameters to be pruned
    amounts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Determining the number of prunable modules
    module_count = sum(1 for _, module in model.named_modules()
                       if hasattr(module, 'weight'))

    parameters_to_prune = []
    
    for module_name, module in model.named_modules():
        # Check, if module carries the attribute 'weight'
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)) and hasattr(module, 'weight') and module.weight is not None:
            parameters_to_prune.append((module, 'weight'))
            
    return amounts, module_count, parameters_to_prune

amounts, module_count, parameters_to_prune = setUpPruning(model)

def globalUnstructuredL1Pruning(amounts, validation_loader, model, parameters_to_prune):
    # Array to store results, one-dimensional as we prune globally
    results_global_unstructured_l1 = np.zeros(len(amounts))
    
    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total_params)
    
    initial_state = copy.deepcopy(model.state_dict())
    
    print("\n########## Global Unstructured L1 Pruning ##########\n")

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
        non_zero_params = sum(p.data.count_nonzero().item()
                              for p in model.parameters())
        pruned_params = total_params - non_zero_params

        print("Non-zero parameters after pruning:", non_zero_params)
        print("Pruned parameters:", pruned_params)

        # Calculate the accuracy after pruning
        correct_predictions, total_samples = correctPred(validation_loader, model)
        accuracy = correct_predictions / total_samples

        print(f"Accuracy: {accuracy:}")

        results_global_unstructured_l1[i] = accuracy

        # Reset the model to its original state (remove pruning)
        for module, _ in parameters_to_prune:
            prune.remove(module, 'weight')
            
        # Restore the initial state of the model
        model.load_state_dict(initial_state)


    correct_predictions, total_samples = correctPred(
        validation_loader, model)
    accuracy = correct_predictions / total_samples

    print(f"Accuracy after Global Unstructured L1: {accuracy:}")
        
    np.save('results_global_unstructured_l1.npy', results_global_unstructured_l1)

def localUnstructuredL1Pruning(module_count, amounts, validation_loader, model):
    # Create an array to save the results
    initial_state = copy.deepcopy(model.state_dict())
    
    results_local_unstructured_l1 = np.zeros((module_count, len(amounts)))
    
    correct_predictions, total_samples = correctPred(
        validation_loader, model)
    accuracy = correct_predictions / total_samples

    print("\n########## Local Unstructured L1 Pruning ##########\n")
    print(f"Accuracy before: {accuracy:}")
    module_idx = 0
    # Loop through different pruning rates
    for module_name, module in model.named_modules():
        
        initial_module_state = copy.deepcopy(module.state_dict())
        # Given modules have weights, we prune them according to the amounts parameter and the l1-norm
        if isinstance(module, (torch.nn.Conv2d)) and hasattr(module, 'weight'):
            for i, pruning_rate in enumerate(amounts):
                prune.l1_unstructured(module, name = 'weight', amount = pruning_rate)

                # Assess the accuracy and store it 
                correct_predictions, total_samples = correctPred(validation_loader, model)
                accuracy = correct_predictions / total_samples

                results_local_unstructured_l1[module_idx, i] = accuracy
                print(f"Module: {module_name}, Pruning Rate: {pruning_rate}, Accuracy: {accuracy}")

                # Reset the model to its original state (remove pruning)
                prune.remove(module, 'weight')
                
                module.load_state_dict(initial_module_state)
                
            module_idx += 1

    model.load_state_dict(initial_state)
    # Save the results
    np.save('results_local_unstructured_l1.npy', results_local_unstructured_l1)

def localUnstructuredRandomPruning(module_count, amounts, validation_loader, model):
    initial_state = copy.deepcopy(model.state_dict())
    
    # Free GPU memory
    torch.cuda.empty_cache()
    
    runs = 2  

    results_local_unstructured_random = np.zeros((module_count, len(amounts), runs + 2))
    
    print("\n########## Local Unstructured Random Pruning ##########\n")

    module_idx = 0

    for module_name, module in model.named_modules():
        # Given modules have weights, we prune them randomly
        if isinstance(module, (torch.nn.Conv2d)) and hasattr(module, 'weight'):
            initial_module_state = copy.deepcopy(module.state_dict())
            for i, pruning_rate in enumerate(amounts):
                accuracies = np.zeros(runs)

                # Simulate k runs
                for k in range(runs):
                    # Apply pruning to the module
                    prune.random_unstructured(
                        module, name='weight', amount=pruning_rate)

                    # Validate after pruning
                    correct_predictions, total_samples = correctPred(
                        validation_loader, model)
                    accuracy = correct_predictions / total_samples

                    accuracies[k] = accuracy

                    # Reset the model to its original state (remove pruning)
                    prune.remove(module, 'weight')
                    
                    module.load_state_dict(initial_module_state)

                # Store mean and standard deviation of accuracies
                results_local_unstructured_random[module_idx,
                                                  i, 0] = accuracies.mean()
                results_local_unstructured_random[module_idx,
                                                  i, 1] = accuracies.std()
                results_local_unstructured_random[module_idx, i, 2:] = accuracies
                
                print(f"Module: {module_name}, Pruning Rate: {pruning_rate}, Accuracy: {accuracies.mean()}")

            module_idx += 1

    model.load_state_dict(initial_state)
    # Save the pruning results
    np.save('results_local_unstructured_random.npy', results_local_unstructured_random)

def LocalStructuredLNPruning(module_count, amounts, validation_loader, model, parameters_to_prune, n):
    # Load the initial state
    initial_state = copy.deepcopy(model.state_dict())

    # Free GPU memory
    torch.cuda.empty_cache()
    
    # Create an array to save the results, 2 = number of dims (0 = Convolutional Layers, 1 = Linear layers)
    results_local_structured_ln = np.zeros((module_count, len(amounts)))

    print(f"\n########## Local Structured L{n} Pruning ##########\n")

    module_index = 0
    for module_name, module in model.named_modules():
        if hasattr(module, 'weight') and isinstance(module, torch.nn.Conv2d):
            # Determine the appropriate dimension for pruning
            initial_module_state = copy.deepcopy(module.state_dict())
            
            for i, pruning_rate in enumerate(amounts):
                prune.ln_structured(
                    module, name='weight', amount=pruning_rate, n=n, dim=0)
                # Assess the accuracy and store it
                correct_predictions, total_samples = correctPred(
                    validation_loader, model)
                accuracy = correct_predictions / total_samples
                results_local_structured_ln[module_index,
                                            i] = accuracy
                print(f"Module: {module_name}, Pruning Rate: {pruning_rate}, Dim: {dim}, Accuracy: {accuracy}")
                # Reset the model to its original state (remove pruning)
                prune.remove(module, 'weight')  
                    
                module.load_state_dict(initial_module_state)

            module_index += 1

    model.load_state_dict(initial_state)
    # Save the results
    np.save(f'results_local_structured_l{n}.npy', results_local_structured_ln)

def localStructuredRandomPruning(module_count, amounts, validation_loader, model, parameters_to_prune, runs=2):
    # Load the initial state
    initial_state = copy.deepcopy(model.state_dict())
    
    # Create an array to save the results
    results_local_structured_random = np.zeros(
        (module_count, len(amounts), runs))

    print("########## Local Structured Random Pruning ##########\n\n")
    
    module_count = 0

    for module_name, module in enumerate(model.named_modules()):
        if hasattr(module, 'weight') and isinstance(module, torch.nn.Conv2d):
            
            initial_module_state = copy.deepcopy(module.state_dict())
            
            for i, pruning_rate in enumerate(amounts):
                # Simulate k runs per pruning rate and module
                for k in range(runs):
                    prune.random_structured(
                        module, name='weight', amount=pruning_rate, dim=0)
                    # Assess the accuracy and store it
                    correct_predictions, total_samples = correctPred(
                        validation_loader, model)
                    accuracy = correct_predictions / total_samples
                    results_local_structured_random[module_count,
                                                    i, k] = accuracy
                    # Reset the model to its original state (remove pruning)
                    prune.remove(module, 'weight')
                        
                    module.load_state_dict(initial_module_state)
                mean = results_local_structured_random[module_count, i, :].mean(
                )
                print(f"Module: {module_name}, Pruning Rate: {pruning_rate}, Accuracy: {mean}")
            
            module_count += 1

    # Save the results
    np.save('results_local_structured_random.npy',
            results_local_structured_random)

def measure_time(pruning_function, *args):
    start_time = time.time()
    pruning_function(*args)
    elapsed_time = time.time() - start_time

    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"{pruning_function.__name__} took {int(hours):02}:{int(minutes):02}:{seconds:05.2f}")

######################## Test Pruning ########################

# Prune the model using global unstructured L1 pruning
# measure_time(globalUnstructuredL1Pruning, amounts, validation_loader, model, parameters_to_prune)

# Prune the model using local unstructured L1 pruning
measure_time(localUnstructuredL1Pruning, module_count, amounts,
              validation_loader, model)

# Prune the model using local unstructured random pruning
measure_time(localUnstructuredRandomPruning, module_count,
             amounts, validation_loader, model)

# Prune the model using local structured L1 pruning
measure_time(LocalStructuredLNPruning, module_count, amounts,
             validation_loader, model, parameters_to_prune, 1)

# Prune the model using local structured random pruning
measure_time(localStructuredRandomPruning, module_count, amounts,
             validation_loader, model, parameters_to_prune, 2)

# Load the results
results_global_unstructured_l1 = np.load('results_global_unstructured_l1.npy')
results_local_unstructured_l1 = np.load('results_local_unstructured_l1.npy')
results_local_unstructured_random = np.load('results_local_unstructured_random.npy')
results_local_structured_l1 = np.load('results_local_structured_l1.npy')
results_local_structured_l2 = np.load('results_local_structured_l2.npy')
results_local_structured_random = np.load('results_local_structured_random.npy')


def plot_and_save_results(amounts, results_global_unstructured_l1, results_local_unstructured_l1,
                          results_local_unstructured_random, results_local_structured_l1,
                          results_local_structured_l2, results_local_structured_random, filename):

    plt.figure(figsize=(10, 6))
    plt.plot(amounts, results_global_unstructured_l1,
             label='Global Unstructured L1')
    plt.plot(amounts, results_local_unstructured_l1.mean(
        axis=0), label='Local Unstructured L1')
    plt.plot(amounts, results_local_unstructured_random.mean(
        axis=(0, 2)), label='Local Unstructured Random')
    plt.plot(amounts, results_local_structured_l1.mean(
        axis=(0, 2)), label='Local Structured L1')
    plt.plot(amounts, results_local_structured_l2.mean(
        axis=(0, 2)), label='Local Structured L2')
    plt.plot(amounts, results_local_structured_random.mean(
        axis=(0, 2, 3)), label='Local Structured Random')
    plt.xlabel('Pruning Rate')
    plt.ylabel('Accuracy')
    plt.title('Pruning Methods')
    plt.legend()
    plt.savefig(filename)
    plt.close()

plot_and_save_results(
    amounts,
    results_global_unstructured_l1,
    results_local_unstructured_l1,
    results_local_unstructured_random,
    results_local_structured_l1,
    results_local_structured_l2,
    results_local_structured_random,
    'pruning_methods_accuracy.png'
)
