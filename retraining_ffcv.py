import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision import models
import torch.nn.utils.prune as prune
import numpy as np
import copy
import json
import time

from ffcv_dataloaders import create_train_loader, create_test_loader

FFCV_PATH = "/mnt/lustre/datasets/ImageNet-ffcv"

# Check if CUDA is available and set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create FFCV data loaders
start_time = time.time()
train_loader = create_train_loader(
    os.path.join(FFCV_PATH, 'train_500_0.50_90.ffcv'),
    num_workers=8,
    batch_size=128,
    distributed=False,
    in_memory=False,
    device=device
)
print(f"Train loader created in {time.time() - start_time} seconds")

start_time = time.time()
val_loader = create_test_loader(
    os.path.join(FFCV_PATH, 'val_500_0.50_90.ffcv'),
    num_workers=8,
    batch_size=128,
    distributed=False,
    in_memory=False,
    device=device
)
print(f"Train loader created in {time.time() - start_time} seconds")

scaler = GradScaler()

# Initialize the GoogLeNet model
model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet',
                       weights='GoogLeNet_Weights.DEFAULT', aux_logits=True)
model = model.to(device)

## Set Hyperparameters
learning_rate = 0.01
epochs = 10

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

print(f"Training for {epochs} epochs with learning rate {learning_rate} and optimizer {optimizer.__class__} and scheduler {scheduler.__class__}")


def validate(model, loader):
    model.eval()
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):  # Check if model has auxiliary outputs
                outputs = outputs[0]
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = correct_predictions / total_samples
    return accuracy

# Function to train the model


def train(model, loader, criterion, optimizer, scheduler, epochs=1, validation_loader=None):
    for epoch in range(epochs):
        model.train()
        if hasattr(model, 'aux1'):
            model.aux1.training = True

        if hasattr(model, 'aux2'):
            model.aux2.training = True

        running_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                if isinstance(outputs, tuple):  # Check if auxiliary outputs are present
                    outputs, aux1, aux2 = outputs  # Unpack main output and auxiliary outputs
                    loss1 = criterion(outputs, labels)
                    loss2 = criterion(aux1, labels)
                    loss3 = criterion(aux2, labels)
                    loss = loss1 + 0.3 * loss2 + 0.3 * loss3
                else:
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)

        scheduler.step()

        # Get current learning rate
        current_lr = scheduler.get_last_lr()[0]

        # Calculate and print training loss
        epoch_loss = running_loss / len(loader)

        if validation_loader:
            accuracy = validate(model, validation_loader)
            print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {epoch_loss}, Learning Rate: {current_lr}, Validation Accuracy: {accuracy}')
        else:
            print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {epoch_loss}, Learning Rate: {current_lr}')


# Prune the model


def count_nonzero_params(model, excluded_modules=[]):
    nonzero_params = 0
    total_params = 0
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and hasattr(module, 'weight'):
            if module_name in excluded_modules:
                continue
            nonzero_params += torch.count_nonzero(module.weight).item()
            total_params += module.weight.numel()
    return nonzero_params, total_params


def correctPred(validation_loader, model):
    correct_predictions = 0
    total_samples = 0
    for images, labels in validation_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
    return correct_predictions, total_samples


def load_pruning_rates(file_path):
    with open(file_path, 'r') as file:
        pruning_rates = json.load(file)
    return pruning_rates


def pruneSpecificLocalStructuredLNPruning(validation_loader, model, n, epochs):
    initial_state = copy.deepcopy(model.state_dict())
    pruning_rates_file = "/home/wichmann/wzz745/Network-Pruning-and-Interpretability/Pruning_Rates/pruning_rates_local_structured_l1.json"
    pruning_rates = load_pruning_rates(pruning_rates_file)

    accuracy = validate(model, validation_loader)

    print(f"\n########## Specific Local Structured L{n} Pruning ##########\n")
    print(f"Accuracy before: {accuracy}")

    excluded_modules = ["conv1.conv", "conv2.conv",
                        "conv3.conv", "aux1.conv.conv", "aux2.conv.conv"]

    # factors = [0.27, 0.55, 0.82, 1.1, 1.37, 1.65, 1.92]
    # avg_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    factors = [0.27, 0.55, 0.82, 1.1]
    avg_rates = [0.1, 0.2, 0.3, 0.4]
    initial_module_states = {}

    index = 0
    for factor in factors:
        print("\n------------------- Pruning Modules -------------------\n")
        for module_name, module in model.named_modules():
            if hasattr(module, 'weight') and isinstance(module, torch.nn.Conv2d):
                if module_name in excluded_modules:
                    continue

                initial_module_states[module_name] = copy.deepcopy(
                    module.state_dict())
                pruning_rate = round(pruning_rates[module_name] * factor, 2)
                prune.ln_structured(module, name='weight',
                                    amount=pruning_rate, n=n, dim=0)
                print(f"Module: {module_name}, Pruning Rate: {pruning_rate}")

        print("\n--------------------------------------------------------\n")
        non_zero_params, total_params = count_nonzero_params(model)
        print(f"Actual Pruning Rate: {1 - non_zero_params / total_params}")

        correct_predictions, total_samples = correctPred(
            validation_loader, model)
        accuracy = correct_predictions / total_samples
        print(f"Avg Pruning Rate: {avg_rates[index]}, Accuracy: {accuracy}")

        train(model, train_loader, criterion, optimizer, scheduler,
              epochs=epochs, validation_loader=validation_loader)

        # remove the pruning masks so they're not retrained again after retraining
        for module_name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d)) and hasattr(module, 'weight'):
                if module_name in excluded_modules:
                    continue
                prune.remove(module, 'weight')

        accuracy = validate(model, validation_loader)
        print(print(f"Accuracy after retraining: {accuracy}"))

        model.load_state_dict(initial_state)
        index += 1


def pruneSpecificLocalUnstructuredL1(validation_loader, model, epochs):
    # Create an array to save the results
    initial_state = copy.deepcopy(model.state_dict())

    # Path to JSON-File with the specific pruning rates
    pruning_rates_file = "/home/wichmann/wzz745/Network-Pruning-and-Interpretability/Pruning_Rates/pruning_rates_local_unstructured_l1.json"
    pruning_rates = load_pruning_rates(pruning_rates_file)

    correct_predictions, total_samples = correctPred(validation_loader, model)
    accuracy = correct_predictions / total_samples

    print("\n########## Specific Local Unstructured L1 Pruning ##########\n")
    print(f"Accuracy before: {accuracy:}")

    excluded_modules = ["conv1.conv", "conv2.conv",
                        "conv3.conv", "aux1.conv.conv", "aux2.conv.conv"]

    # finetuning of the pruning rates
    factors = [0.13, 0.26, 0.4, 0.53, 0.66, 0.79, 0.92]
    avg_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    non_zero_params, total_params = count_nonzero_params(model)
    print(
        f"Non-zero params before Pruning: {non_zero_params}, Total params: {total_params}")

    index = 0
    for factor in factors:
        print("\n------------------- Pruning Modules -------------------\n")
        for module_name, module in model.named_modules():

            # Given modules have weights, we prune them according to the amounts parameter and the l1-norm
            if isinstance(module, (torch.nn.Conv2d)) and hasattr(module, 'weight'):

                if module_name in excluded_modules:
                    continue

                # get pruning rate from json file and round to 2 decimal places
                pruning_rate = round(
                    pruning_rates[module_name] * factor, 2)

                prune.l1_unstructured(
                    module, name='weight', amount=pruning_rate)

                print(f"Module: {module_name}, Pruning Rate: {pruning_rate}")

        print("\n--------------------------------------------------------\n")
        print(f"\n Avg Pruning Rate: {avg_rates[index]} \n")
        non_zero_params, total_params = count_nonzero_params(model)

        print(f"Actual Pruning Rate: {1 - non_zero_params / total_params}")
        # Assess the accuracy and store it
        correct_predictions, total_samples = correctPred(
            validation_loader, model)
        accuracy = correct_predictions / total_samples

        print("Average Pruning Accuracy: ",
              avg_rates[index], " Accuracy: ", accuracy)

        # retrain the model
        train(model, train_loader, criterion, optimizer,
              scheduler, epochs=epochs, validation_loader=validation_loader)

        # after retraining, remove the pruning masks so they're not retrained again
        for module_name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d)) and hasattr(module, 'weight'):
                if module_name in excluded_modules:
                    continue
                prune.remove(module, 'weight')

        model.eval()
        # validate accuracy after pruning
        accuracy_retrained = validate(model, validation_loader)
        print("Accuracy after retraining:", accuracy_retrained)

        model.load_state_dict(initial_state)
        index += 1


pruneSpecificLocalStructuredLNPruning(val_loader, model, 1, epochs)
#pruneSpecificLocalUnstructuredL1(val_loader, model, epochs)
print("Finished pruning, retraining, and evaluation.")


