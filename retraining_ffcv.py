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
import wandb


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
print(f"Train loader created in {time.time() - start_time} seconds", flush=True)

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

# Set Hyperparameters
learning_rate = 0.001
epochs = 10

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=learning_rate,
# momentum=0.9, weight_decay=1e-4)

optimizer = optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

print(f"Training for {epochs} epochs with learning rate {learning_rate} and optimizer {optimizer.__class__.__name__} and scheduler {scheduler.__class__.__name__}")

wandb.login(key="ea551b0198dda65a9f311d0ea5d6eaa6f41b1d4a")
run = wandb.init(
    # Set the project where this run will be logged
    project="epic",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "architecture": "GoogleNet",
        "dataset": "ImageNet",
        "epochs": epochs,
        "optimizer": optimizer.__class__.__name__,
        "scheduler": scheduler.__class__.__name__,
    },
)


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
        running_loss_without_aux = 0.0
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

            running_loss_without_aux += loss1.item()
            running_loss += loss.item()

        scheduler.step()

        # Get current learning rate
        current_lr = scheduler.get_last_lr()[0]

        # Calculate and print training loss
        epoch_loss = running_loss / len(loader)
        epoch_loss_without_aux = running_loss_without_aux / len(loader)

        if validation_loader:
            accuracy = validate(model, validation_loader)
            print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {epoch_loss}, Training Loss w/o Aux: {epoch_loss_without_aux}, Learning Rate: {current_lr}, Validation Accuracy: {accuracy}', flush=True)
            wandb.log({"accuracy": accuracy, "training loss": epoch_loss, "training loss w/o aux": {
                      epoch_loss_without_aux}, "learning rate": current_lr, "epoch": epoch+1})
        else:
            print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {epoch_loss}, Learning Rate: {current_lr}')


# Prune the model

def removePruningMasks(model, excluded_modules=[]):
    # after retraining, remove the pruning masks so they're not retrained again
    for module_name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d)) and hasattr(module, 'weight'):
            if module_name in excluded_modules:
                continue
            prune.remove(module, 'weight')


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


def load_pruning_rates(file_path):
    with open(file_path, 'r') as file:
        pruning_rates = json.load(file)
    return pruning_rates


def pruneSpecificLocalStructuredLNPruning(validation_loader, model, n):
    initial_state = copy.deepcopy(model.state_dict())
    pruning_rates_file = "/home/wichmann/wzz745/Network-Pruning-and-Interpretability/Pruning_Rates/pruning_rates_local_structured_l1.json"
    pruning_rates = load_pruning_rates(pruning_rates_file)

    accuracy = validate(model, validation_loader)

    print(f"\n########## Specific Local Structured L{n} Pruning ##########\n")
    print(f"Accuracy before: {accuracy}")

    excluded_modules = ["conv1.conv", "conv2.conv",
                        "conv3.conv", "aux1.conv.conv", "aux2.conv.conv"]

    excluded_modules = ["conv1.conv", "conv2.conv",
                        "conv3.conv", "aux1.conv.conv", "aux2.conv.conv"]

    rates = [0.4]
    epochen = [50]
    for epochs in epochen:

        for rate in rates:
            accuracy = validate(model, validation_loader)
            print("Accuracy before: ", accuracy)

            print(f"\n------------------- Pruning Modules with {rate} -------------------\n")
            for module_name, module in model.named_modules():
                if hasattr(module, 'weight') and isinstance(module, torch.nn.Conv2d):

                    if module_name in excluded_modules:
                        continue

                    # Prune the module
                    prune.ln_structured(
                        module, name='weight', amount=rate, n=n, dim=0)

            print("\n--------------------------------------------------------\n")

            non_zero_params, total_params = count_nonzero_params(model)
            print(f"Actual Pruning Rate: {1 - non_zero_params / total_params}")

            # Assess the accuracy and store it
            accuracy = validate(model, validation_loader)
            print(f"Accuracy after pruning every module with {rate}: ", accuracy)

            # Reset learning rate (rewinding) and use Adam optimizer
            # optimizer = optim.Adam(
            #     model.parameters(), lr=learning_rate, weight_decay=0.0001)
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            train(model, train_loader, criterion, optimizer,
                  scheduler, epochs, validation_loader)

            accuracy_retrained = validate(model, validation_loader)
            print("Accuracy after retraining:", accuracy_retrained)
            print("removing pruning masks ...")
            removePruningMasks(model, excluded_modules)

            # Save the final pruned and retrained model
            optimizer_name = optimizer.__class__.__name__
            torch.save(model, f'pruned_{rate}_local_structured_{optimizer_name}_retrained_{epochs}_epochs_model.pth')
            print(f"Final pruned and retrained model saved as pruned_{rate}_local_structured_{optimizer_name}_retrained_{epochs}_epochs_model.pth")

            print("\nResetting the model to the initial state ...")
            model.load_state_dict(initial_state)


def pruneSpecificLocalUnstructuredL1(validation_loader, model, epochs):
    # Create an array to save the results
    initial_state = copy.deepcopy(model.state_dict())

    # Path to JSON-File with the specific pruning rates
    pruning_rates_file = "/home/wichmann/wzz745/Network-Pruning-and-Interpretability/Pruning_Rates/pruning_rates_local_unstructured_l1.json"
    pruning_rates = load_pruning_rates(pruning_rates_file)

    accuracy = validate(model, validation_loader)

    print("\n########## Specific Local Unstructured L1 Pruning ##########\n")
    print(f"Accuracy before: {accuracy:}")

    excluded_modules = ["conv1.conv", "conv2.conv",
                        "conv3.conv", "aux1.conv.conv", "aux2.conv.conv"]

    # finetuning of the pruning rates
    factors = [0.26, 0.53, 0.79, 1.06]
    # factors = [0.13, 0.26, 0.4, 0.53, 0.66, 0.79, 0.92]
    # avg_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    avg_rates = [0.2, 0.4, 0.6, 0.8]

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
        accuracy = validate(model, validation_loader)

        print("Average Pruning Accuracy: ",
              avg_rates[index], " Accuracy: ", accuracy)

        # reset learning rate (rewinding)
        initial_lr = 0.001  # Store the initial learning rate
        optimizer = optim.SGD(model.parameters(), lr=initial_lr,
                              momentum=0.9, weight_decay=0.0001)
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


def pruneSpecificLocalStructuredLNPruningSuccessively(validation_loader, model, n):
    initial_state = copy.deepcopy(model.state_dict())
    # Path to JSON-File with the specific pruning rates
    pruning_rates_file = "/home/wichmann/wzz745/Network-Pruning-and-Interpretability/Pruning_Rates/pruning_rates_local_structured_l1.json"
    pruning_rates = load_pruning_rates(pruning_rates_file)

    accuracy = validate(model, validation_loader)

    print(f"\n########## Specific Local Structured L{n} Pruning Successively ##########\n")
    print(f"Accuracy before: {accuracy:}")

    excluded_modules = ["conv1.conv", "conv2.conv",
                        "conv3.conv", "aux1.conv.conv", "aux2.conv.conv"]

    factor = 0.55
    avg_rate = 0.2

    print("\n------------------- Pruning Modules with 0.8 -------------------\n")
    for module_name, module in model.named_modules():
        if hasattr(module, 'weight') and isinstance(module, torch.nn.Conv2d):

            if module_name in excluded_modules:
                continue

            # get pruning rate from json file and round to 2 decimal places
            pruning_rate = 0.8
            # Prune the module
            prune.ln_structured(
                module, name='weight', amount=pruning_rate, n=n, dim=0)

            print(f"Module: {module_name}, Pruning Rate: {pruning_rate}")

    print("\n--------------------------------------------------------\n")

    non_zero_params, total_params = count_nonzero_params(model)
    print(f"Actual Pruning Rate: {1 - non_zero_params / total_params}")
    # Assess the accuracy and store it
    accuracy = validate(model, validation_loader)
    print("Accuracy after pruning every module with 0.6: ", accuracy)
    # reset learning rate (rewinding)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    train(model, train_loader, criterion, optimizer,
          scheduler, 40, validation_loader)
    accuracy_retrained = validate(model, validation_loader)
    print("Accuracy after retraining:", accuracy_retrained)
    print("removing pruning masks ...")
    removePruningMasks(model, excluded_modules)

    # Save the final pruned and retrained model
    torch.save(model.state_dict(),
               'pruned_local_structured_retrained_model.pth')
    print("Final pruned and retrained model saved as pruned_local_structured_retrained_model.pth")

    # Resetting the model to the initial state
    print("\nResetting the model to the initial state ...")
    model.load_state_dict(initial_state)
    accuracy = validate(model, validation_loader)
    print("Accuracy before:", accuracy)

    print("\n------------------- Pruning Modules specific iteratively with avg 0.2 -------------------\n")

    index = 0
    for i in range(4):
        print("\n------------------- Pruning Modules -------------------\n")
        for module_name, module in model.named_modules():
            if hasattr(module, 'weight') and isinstance(module, torch.nn.Conv2d):

                if module_name in excluded_modules:
                    continue

                # get pruning rate from json file and round to 2 decimal places
                pruning_rate = round(
                    pruning_rates[module_name] * factor, 2)
                # Prune the module
                prune.ln_structured(
                    module, name='weight', amount=pruning_rate, n=n, dim=0)

                print(f"Module: {module_name}, Pruning Rate: {pruning_rate}")

        print("\n--------------------------------------------------------\n")

        # Assess the accuracy and store it
        accuracy = validate(model, validation_loader)

        print("Relative Pruning Rate: ",
              avg_rate)

        if index == 0:
            absolute_pruning_rate = avg_rate
        else:
            absolute_pruning_rate = (
                1 - (1 - absolute_pruning_rate) * (1 - avg_rate))

        print("Absolute Pruning Rate: ",absolute_pruning_rate)
        non_zero_params, total_params = count_nonzero_params(model)
        print(f"Actual Pruning Rate: {1 - non_zero_params / total_params}")
        print("Accuracy: ", accuracy)

        # reset learning rate (rewinding)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                              momentum=0.9, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        # retrain the model
        train(model, train_loader, criterion, optimizer,
              scheduler, epochs, validation_loader)

        # # after retraining, remove the pruning masks so they're not retrained again
        # for module_name, module in model.named_modules():
        #     if isinstance(module, (torch.nn.Conv2d)) and hasattr(module, 'weight'):
        #         if module_name in excluded_modules:
        #             continue
        #         prune.remove(module, 'weight')

        model.eval()
        # validate accuracy after pruning
        accuracy_retrained = validate(model, validation_loader)
        print("Accuracy after retraining:", accuracy_retrained)

        index += 1

    # Save the final pruned and retrained model
    torch.save(model.state_dict(
    ), 'pruned_local_structured_specific_iterative_20_retrained_model.pth')
    print("Final pruned and retrained model saved as pruned_local_structured_specific_iterative_20_retrained_model.pth")

    # Removing Pruning Masks
    print("removing pruning masks ...")
    removePruningMasks(model, excluded_modules)

    # Resetting the model to the initial state
    print("\nResetting the model to the initial state ...")
    model.load_state_dict(initial_state)
    accuracy = validate(model, validation_loader)
    print("Accuracy before:", accuracy)

    print("\n------------------- Pruning Modules iteratively with 0.2 -------------------\n")

    index = 0
    for i in range(4):
        print("\n------------------- Pruning Modules -------------------\n")
        for module_name, module in model.named_modules():
            if hasattr(module, 'weight') and isinstance(module, torch.nn.Conv2d):

                if module_name in excluded_modules:
                    continue

                # get pruning rate from json file and round to 2 decimal places
                pruning_rate = 0.2
                # Prune the module
                prune.ln_structured(
                    module, name='weight', amount=pruning_rate, n=n, dim=0)

                print(f"Module: {module_name}, Pruning Rate: {pruning_rate}")

        print("\n--------------------------------------------------------\n")

        # Assess the accuracy and store it
        accuracy = validate(model, validation_loader)

        print("Relative Pruning Rate: ",
              avg_rate)

        if index == 0:
            absolute_pruning_rate = avg_rate
        else:
            absolute_pruning_rate = (
                1 - (1 - absolute_pruning_rate) * (1 - avg_rate))

        print("Absolute Pruning Rate: ",absolute_pruning_rate)
        non_zero_params, total_params = count_nonzero_params(model)
        print(f"Actual Pruning Rate: {1 - non_zero_params / total_params}")
        print("Accuracy: ", accuracy)

        # reset learning rate (rewinding)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                              momentum=0.9, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        # retrain the model
        train(model, train_loader, criterion, optimizer,
              scheduler, epochs, validation_loader)

        model.eval()
        # validate accuracy after pruning
        accuracy_retrained = validate(model, validation_loader)
        print("Accuracy after retraining:", accuracy_retrained)

        index += 1

    # Removing Pruning Masks
    print("removing pruning masks ...")
    removePruningMasks(model, excluded_modules)

    accuracy_retrained = validate(model, validation_loader)
    print("Accuracy after retraining:", accuracy_retrained)

    # Save the final pruned and retrained model
    torch.save(model.state_dict(),
               'pruned_local_structured_iterative_20_retrained_model.pth')
    print("Final pruned and retrained model saved as pruned_local_structured_iterative_20_retrained_model.pth")


def globalUnstructuredL1Pruning(validation_loader, model):

    initial_state = copy.deepcopy(model.state_dict())
    # Accuracy before pruning
    accuracy = validate(model, validation_loader)

    # Exclude the following modules from pruning
    excluded_modules = ["conv1.conv", "conv2.conv",
                        "conv3.conv", "aux1.conv.conv", "aux2.conv.conv"]

    print("\n########## Global Unstructured L1 Pruning Iteratively ##########\n")
    print(f"Accuracy before: {accuracy:}")

    # Count non-zero trainable parameters before pruning
    non_zero_params, total_params = count_nonzero_params(model)
    print(
        f"Non-zero params before Pruning: {non_zero_params}, Total params: {total_params}")

    # Parameters to prune
    parameters_to_prune = []
    for module_name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)) and hasattr(module, 'weight') and module.weight is not None:
            if module_name in excluded_modules:
                continue
            parameters_to_prune.append((module, 'weight'))

    # Define the pruning rate
    rates = [0.2, 0.4, 0.6, 0.8]

    # Prune iteratively with pruning rate of 0.4
    for rate in rates:
        accuracy_before = validate(model, validation_loader)
        print(f"Accuracy before: {accuracy_before:}")
        print(f"Pruning Rate: {rate}")

        print(
            f"\n------------------- Pruning Globally with {rate} -------------------\n")
        # Prune the model
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=rate,
        )
        print("\n--------------------------------------------------------\n")

        # Assess the accuracy and store it
        accuracy = validate(model, validation_loader)

        non_zero_params, total_params = count_nonzero_params(model)
        print(f"Actual Pruning Rate: {1 - non_zero_params / total_params}")
        print("Accuracy: ", accuracy)

        # reset learning rate (rewinding)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                              momentum=0.9, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        # retrain the model
        train(model, train_loader, criterion, optimizer,
              scheduler, epochs, validation_loader)

        removePruningMasks(model, excluded_modules)

        model.eval()

        # validate accuracy after pruning
        accuracy_retrained = validate(model, validation_loader)
        print("Accuracy after retraining:", accuracy_retrained)

        # Reset
        model.load_state_dict(initial_state)


def prune_channels(module, channels_to_prune):
    # Create a mask with the same shape as the module weight
    mask = torch.ones_like(module.weight)
    # Set the selected input channels to zero in the mask
    for channel in channels_to_prune:
        mask[:, channel, :, :] = 0
    # Apply the mask to the module
    module.weight.data.mul_(mask)


def prune_specific_local_connection_sparsity(validation_loader, model):

    initial_state = copy.deepcopy(model.state_dict())
    accuracy = validate(model, validation_loader)

    print(f"\n########## Specific Local Connection Sparsity Pruning ##########\n")
    print(f"Accuracy before: {accuracy:.4f}")

    excluded_modules = ["conv1.conv", "conv2.conv",
                        "conv3.conv", "aux1.conv.conv", "aux2.conv.conv"]

    rates = [0.8]
    epochen = [50]

    for epochs in epochen:
        for rate in rates:
            accuray_before = validate(model, validation_loader)
            print(f"Accuracy before: {accuray_before:.4f}")
            print(
                f"\n------------------- Pruning Input Channels of Modules with {rate} -------------------\n")
            for module_name, module in model.named_modules():
                if hasattr(module, 'weight') and isinstance(module, torch.nn.Conv2d):

                    if module_name in excluded_modules:
                        continue

                    weight = module.weight.detach().cpu().numpy()
                    num_input_channels = weight.shape[1]
                    num_prune = int(num_input_channels * rate)

                    # Zufälliges Entfernen von Verbindungen (Input Channels)
                    channels_to_prune = np.random.choice(
                        num_input_channels, num_prune, replace=False)

                    # Prune the selected input channels
                    prune_channels(module, channels_to_prune)

                    print(f"Module: {module_name}, Pruned Input Channels: {len(channels_to_prune) / num_input_channels}")

            print("\n--------------------------------------------------------\n")

            non_zero_params, total_params = count_nonzero_params(model)
            print(f"Actual Pruning Rate: {1 - non_zero_params / total_params:.4f}")

            # Assess the accuracy and store it
            accuracy = validate(model, validation_loader)
            print(f"Accuracy after pruning every module with {rate}: {accuracy:.4f}")

            # Reset learning rate (rewinding) and use Adam optimizer
            # optimizer = optim.Adam(
            #    model.parameters(), lr=learning_rate, weight_decay=0.0001)
            optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                                  momentum=0.9, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            train(model, train_loader, criterion, optimizer,
                  scheduler, epochs, validation_loader)

            accuracy_retrained = validate(model, validation_loader)
            print(f"Accuracy after retraining: {accuracy_retrained:.4f}")
            print("Removing pruning masks ...")

            # Save the final pruned and retrained model
            # Save the final pruned and retrained model
            optimizer_name = optimizer.__class__.__name__
            torch.save(model, f'pruned_{rate}_connection_sparsity_{optimizer_name}_retrained_{epochs}_epochs_model.pth')

            print("\nResetting the model to the initial state ...")
            model.load_state_dict(initial_state)


# pruneSpecificLocalUnstructuredL1(val_loader, model, epochs)
# prune_specific_local_connection_sparsity(val_loader, model)
pruneSpecificLocalStructuredLNPruning(val_loader, model, 1)
# pruneSpecificLocalStructuredLNPruningSuccessively(val_loader, model, 1)
# globalUnstructuredL1PruningIteratively(val_loader, model)
print("Finished pruning, retraining, and evaluation.")
