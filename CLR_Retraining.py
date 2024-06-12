# all required imports
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.datasets as datasets
import urllib
from PIL import Image
from torchvision import transforms
import torch.nn.utils.prune as prune
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import numpy as np

# Load the GoogleNet model with auxiliary layers enabled
model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', weights='GoogLeNet_Weights.DEFAULT', aux_logits=True)
model.eval()

# URL for the imagenet_classes.txt file
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
filename = "imagenet_classes.txt"

# Download the imagenet_classes.txt file
urllib.request.urlretrieve(url, filename)

# Load and preprocess the example image
url_image, filename_image = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
urllib.request.urlretrieve(url_image, filename_image)

input_image = Image.open(filename_image)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

# Create the validation_loader
validation_set = datasets.ImageFolder('/Users/emiliamosthaf/Documents/viertes semester/teamprojekt/imagenette2-320',
                                      transform=transforms.Compose([
                                          transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225]),
                                      ]))

# Use a smaller subset for validation
validation_indices = range(100)
validation_sampler = SubsetRandomSampler(validation_indices)
validation_loader = DataLoader(validation_set, batch_size=128, sampler=validation_sampler, num_workers=0)

# Create the training_loader
train_set = datasets.ImageFolder('/Users/emiliamosthaf/Documents/viertes semester/teamprojekt/imagenette2-320',
                                 transform=transforms.Compose([
                                     transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225]),
                                 ]))
train_loader = DataLoader(train_set, batch_size=128,sampler=validation_sampler , num_workers=0)

# Check if train_loader has data
if len(train_loader) == 0:
    raise ValueError("The training data loader is empty. Please check your training dataset path and contents.")

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()

# Pruning amount for the loop
amounts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Create a multidimensional array that can store all the values
results_global_unstructured_l1 = np.zeros(len(amounts))

# Function to apply CLR
def train_model_with_clr(model, train_loader, criterion, base_lr, max_lr, steps_per_epoch, epochs):
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=max(1, steps_per_epoch//2), mode='triangular')
    
    model.train()  # Ensure the model is in training mode
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            if torch.cuda.is_available():
                data, target = data.to('cuda'), target.to('cuda')
            
            optimizer.zero_grad()
            output, aux_output1, aux_output2 = model(data)
            loss = criterion(output, target) + 0.3 * criterion(aux_output1, target) + 0.3 * criterion(aux_output2, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
    
    model.eval()

# Loop through different pruning rates
for i, pruning_rate in enumerate(amounts):
    print(f"Pruning Rate: {pruning_rate}")

    # Apply global unstructured L1 pruning to the entire model
    parameters_to_prune = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            parameters_to_prune.append((module, 'weight'))

    # Prune the model using L1 pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_rate,
    )

    # Train the model using CLR after pruning
    if torch.cuda.is_available():
        model = model.to('cuda')
    
    train_model_with_clr(model, train_loader, criterion, base_lr=1e-5, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=5)

    # Validate accuracy after pruning and retraining
    correct_predictions = 0
    for images, labels in validation_loader:
        if torch.cuda.is_available():
            images = images.to('cuda')
            labels = labels.to('cuda')

        with torch.no_grad():
            output = model(images)

        _, prediction = torch.max(output, 1)
        correct_predictions += (prediction == labels).sum().item()

    # Calculate the accuracy for the validation set after L1 pruning and retraining
    accuracy_after_pruning_global_unstructured_l1 = correct_predictions / len(validation_indices)  # Use the length of the subset
    print("Accuracy after L1 pruning and retraining:", accuracy_after_pruning_global_unstructured_l1)

    results_global_unstructured_l1[i] = accuracy_after_pruning_global_unstructured_l1

    # Reset the model to its original state (remove pruning)
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')

np.save('results_globalUnstructured_l1.npy', results_global_unstructured_l1)
