import torch
import torchvision
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.datasets as datasets
import copy
from torchvision.datasets import ImageNet
import os
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.imagenet import load_meta_file
import json


class HackyImageNet(ImageNet):

    def init(self, root: str, devkit_loc="/mnt/qb/datasets/ImageNet2012/", split: str = 'train', transform=None, download=False, **kwargs):
        if download:
            raise NotImplementedError(
                "Automatic download of the ImageNet dataset is not supported.")
        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train", "val"))
        self.devkit_loc = devkit_loc

        wnid_to_classes = load_meta_file(self.devkit_loc)[0]

        super(ImageNet, self).init(self.split_folder, **kwargs)
        self.transform = transform
        self.root = root
        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}
# Load the GoogleNet model


# load the GoogleNet model
# model = models.googlenet(aux_logits=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', weights='googlenet', aux_logits=True)
model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet',
                       weights='GoogLeNet_Weights.DEFAULT', aux_logits=True)
model.train()  # Set the model to training mode


# Enable auxiliary heads for training if they exist
if hasattr(model, 'aux1'):
    model.aux1.training = True
if hasattr(model, 'aux2'):
    model.aux2.training = True

# move model to GPU if available
if torch.cuda.is_available():
    model = model.cuda()

# validation loop for the ffcv imagenet
validation_transformation = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_transformation = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.1),
    transforms.RandomRotation(degrees=30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# Beispiel zur Verwendung:
root_path = "/scratch_local/datasets/ImageNet2012"

batch_size = 256

validation_set = HackyImageNet(
    root=root_path, split='val', transform=validation_transformation)
validation_loader = torch.utils.data.DataLoader(
    validation_set, batch_size=batch_size, shuffle=False, num_workers=8)

train_set = HackyImageNet(
    root=root_path, split='train', transform=train_transformation)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=8)

# function to validate the model


def validate(model, loader):
    model.eval()
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # during validation gradient is not computed to save storage space
        for images, labels in loader:
            if torch.cuda.is_available():
                images = images.to('cuda')
                labels = labels.to('cuda')

            outputs = model(images)
            if isinstance(outputs, tuple):  # check if model has auxiliary outputs
                outputs = outputs[0]

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = (correct_predictions / total_samples) * 100
    return accuracy

# function to train the model


def train(model, loader, criterion, optimizer, scheduler, epochs=1, validation_loader=None):
    # Set the model to training mode
    model.train()

    # Enable auxiliary heads for training if they exist
    if hasattr(model, 'aux1'):
        model.aux1.training = True
    if hasattr(model, 'aux2'):
        model.aux2.training = True

    print("Starting training...")
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in loader:
            if torch.cuda.is_available():
                images = images.to('cuda')
                labels = labels.to('cuda')

            optimizer.zero_grad()

            outputs = model(images)

            if isinstance(outputs, tuple):  # Check if auxiliary outputs are present
                outputs, aux1, aux2 = outputs  # Unpack main output and auxiliary outputs
                loss1 = criterion(outputs, labels)
                loss2 = criterion(aux1, labels)
                loss3 = criterion(aux2, labels)

                loss = loss1 + 0.3 * loss2 + 0.3 * loss3
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        # Get current learning rate
        current_lr = scheduler.get_last_lr()[0]

        # Calculate and print training loss
        epoch_loss = running_loss / len(loader)
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {epoch_loss}, Learning Rate: {current_lr}')


# calculate accuracy for the validation set before pruning
accuracy_before_pruning = validate(model, validation_loader)
print("Accuracy before pruning:", accuracy_before_pruning)

# -----------------------------------------------------------------------------------------------
# L1 UNSTRUCTURED PRUNING (GLOBAL)
# ------------------------------------------------------------------------------------------------

initial_state = copy.deepcopy(model.state_dict())

# pruning amount for the loop
# amounts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
amounts = [0.3, 0.5, 0.7]

# initialize results dictionary
results_l1_unstructured_test = np.zeros(len(amounts))
epochs = 10

criterion = nn.CrossEntropyLoss()

initial_lr = 0.001  # Store the initial learning rate
optimizer = optim.SGD(model.parameters(), lr=initial_lr,
                      momentum=0.9, weight_decay=0.0001)
# optimizer = optim.SGD(model.parameters(), lr=0.1)

# scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

best_val_loss = float('inf')
patience = 5
patience_counter = 0

# prune the model


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


def reset_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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

# Funktion, um die JSON-Datei zu laden


def load_pruning_rates(file_path):
    with open(file_path, 'r') as file:
        pruning_rates = json.load(file)
    return pruning_rates


def pruneSpecificLocalUnstructuredL1(validation_loader, model):
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
    factors = [0.4, 0.53, 0.66, 0.79, 0.92]
    avg_rates = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    non_zero_params, total_params = count_nonzero_params(model)
    print(f"Non-zero params before Pruning: {non_zero_params}, Total params: {total_params}")

    index = 0
    for factor in factors:
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

        print(f"\n Avg Pruning Rate: {avg_rates[index]} \n")
        non_zero_params, total_params = count_nonzero_params(model)
        
        print(f"Actual Pruning Rate: {1 - non_zero_params / total_params}")
        # Assess the accuracy and store it
        correct_predictions, total_samples = correctPred(
            validation_loader, model)
        accuracy = correct_predictions / total_samples

        print("Average Pruning Accuracy: ", avg_rates[index], " Accuracy: ", accuracy)

        # retrain the model
        train(model, train_loader, criterion, optimizer,
              scheduler, epochs, validation_loader)

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


def pruneSpecificLocalUnstructuredL1Successively(validation_loader, model):

    # Path to JSON-File with the specific pruning rates
    pruning_rates_file = "/home/wichmann/wzz745/Network-Pruning-and-Interpretability/Pruning_Rates/pruning_rates_local_unstructured_l1.json"
    pruning_rates = load_pruning_rates(pruning_rates_file)

    correct_predictions, total_samples = correctPred(validation_loader, model)
    accuracy = correct_predictions / total_samples

    print("\n########## Specific Local Unstructured L1 Pruning Successively ##########\n")
    print(f"Accuracy before: {accuracy:}")

    excluded_modules = ["conv1.conv", "conv2.conv",
                        "conv3.conv", "aux1.conv.conv", "aux2.conv.conv"]

    # finetuning of the pruning rates
    factors = [0.4, 0.66, 0.92]
    avg_rates = [0.3, 0.5, 0.7]
    
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
        
        # Assess the accuracy and store it
        correct_predictions, total_samples = correctPred(
            validation_loader, model)
        accuracy = correct_predictions / total_samples

        print("Relative Pruning Rate: ",
              avg_rates[index])

        if index == 0:
            absolute_pruning_rate = avg_rates[index]
        else:
            absolute_pruning_rate = (1 - (1 - absolute_pruning_rate) * avg_rates[index] )

        print("Absolute Pruning Rate: ",
              absolute_pruning_rate)
        # reset learning rate (rewinding)
        non_zero_params, total_params = count_nonzero_params(model)
        print(f"Actual Pruning Rate: {1 - non_zero_params / total_params}")
        print("Accuracy: ", accuracy)
        
        # rewinding learning_rate to 0.001
        reset_learning_rate(optimizer, initial_lr)
                              momentum=0.9, weight_decay=0.0001)
        # retrain the model
        train(model, train_loader, criterion, optimizer,
              scheduler, epochs, validation_loader)

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

        index += 1


def pruneSpecificLocalStructuredLNPruning(validation_loader, model, n):
    
    # Saving initial state of the model
    initial_state = copy.deepcopy(model.state_dict())
    
    # Path to JSON-File with the specific pruning rates
    pruning_rates_file = "/home/wichmann/wzz745/Network-Pruning-and-Interpretability/Pruning_Rates/pruning_rates_local_structured_l1.json"
    pruning_rates = load_pruning_rates(pruning_rates_file)

    correct_predictions, total_samples = correctPred(validation_loader, model)
    accuracy = correct_predictions / total_samples

    print(f"\n########## Specific Local Structured L{n} Pruning ##########\n")
    print(f"Accuracy before: {accuracy:}")

    excluded_modules = ["conv1.conv", "conv2.conv",
                        "conv3.conv", "aux1.conv.conv", "aux2.conv.conv"]

    factors = [0.27, 0.55, 0.82, 1.1, 1.37, 1.65, 1.92]
    avg_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    index = 0
    for factor in factors:
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
        print(f"Avg Pruning Rate: {avg_rates[index]}")
        
        non_zero_params, total_params = count_nonzero_params(model)

        print(f"Actual Pruning Rate: {1 - non_zero_params / total_params}")
        # Assess the accuracy and store it
        correct_predictions, total_samples = correctPred(
            validation_loader, model)
        accuracy = correct_predictions / total_samples

        print("Average Pruning Accuracy: ",
              avg_rates[index], " Accuracy: ", accuracy)
        
        # reset learning rate (rewinding)
        initial_lr = 0.001  # Store the initial learning rate
        optimizer = optim.SGD(model.parameters(), lr=initial_lr,
                      momentum=0.9, weight_decay=0.0001)
        
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

        model.load_state_dict(initial_state)
        index += 1

    
def pruneSpecificLocalStructuredLNPruningSuccessively(validation_loader, model, n):
    # Path to JSON-File with the specific pruning rates
    pruning_rates_file = "/home/wichmann/wzz745/Network-Pruning-and-Interpretability/Pruning_Rates/pruning_rates_local_structured_l1.json"
    pruning_rates = load_pruning_rates(pruning_rates_file)

    correct_predictions, total_samples = correctPred(validation_loader, model)
    accuracy = correct_predictions / total_samples

    print(f"\n########## Specific Local Structured L{n} Pruning Successively ##########\n")
    print(f"Accuracy before: {accuracy:}")

    excluded_modules = ["conv1.conv", "conv2.conv",
                        "conv3.conv", "aux1.conv.conv", "aux2.conv.conv"]

    factor = 0.27
    avg_rate = 0.1

    index = 0
    for i in range(7):
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
        correct_predictions, total_samples = correctPred(
            validation_loader, model)
        accuracy = correct_predictions / total_samples

        print("Relative Pruning Rate: ",
              avg_rate)

        if index == 0:
            absolute_pruning_rate = avg_rate
        else:
            absolute_pruning_rate = (
                1 - (1 - absolute_pruning_rate) * (1 - avg_rate))

        print("Absolute Pruning Rate: ",
              absolute_pruning_rate)
        non_zero_params, total_params = count_nonzero_params(model)
        print(f"Actual Pruning Rate: {1 - non_zero_params / total_params}")
        print("Accuracy: ", accuracy)

        # Rewinding learning rate before retraining
        reset_learning_rate(optimizer, initial_lr)
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

#pruneSpecificLocalUnstructuredL1(validation_loader, model)

#pruneSpecificLocalUnstructuredL1Successively(validation_loader, model)

#pruneSpecificLocalStructuredLNPruning(validation_loader, model, 1)

pruneSpecificLocalStructuredLNPruningSuccessively(validation_loader, model, 1)

print("Finished pruning, retraining, and evaluation.")
