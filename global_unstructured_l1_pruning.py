# all required imports
import torch
import torchvision.datasets as datasets
import urllib
from PIL import Image
from torchvision import transforms
import json
import torch.nn.utils.prune as prune
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pandas as pd

# Load the GoogleNet model
model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', weights='GoogLeNet_Weights.DEFAULT')
model.eval()

# URL for the imagenet_classes.txt file
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
filename = "imagenet_classes.txt"

# Download the imagenet_classes.txt file
urllib.request.urlretrieve(url, filename)

# Load and preprocess the example image
url_image, filename_image = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try:
    urllib.URLopener().retrieve(url_image, filename_image)
except:
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
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=128,
                                                sampler=torch.utils.data.SubsetRandomSampler(range(100)),
                                                num_workers=0)

## Global unstructured pruning for the whole model

# Pruning amount for the loop
amounts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# create a multidimensional array that can store all the values
results_global_unstructured_l1 = np.zeros(len(amounts))

# Loop through different pruning rates
for i, pruning_rate in enumerate(amounts):
    print(f"Pruning Rate: {pruning_rate}")

    # Apply global unstructured L1 pruning to the entire model

    # Collect all parameters in the model that can be pruned
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

    # Validate accuracy after pruning
    correct_predictions = 0
    for images, labels in validation_loader:
        if torch.cuda.is_available():
            images = images.to('cuda')
            labels = labels.to('cuda')

        with torch.no_grad():
            output = model(images)

        _, prediction = torch.max(output, 1)
        correct_predictions += (prediction == labels).sum().item()

    # Calculate the accuracy for the validation set after L1 pruning
    accuracy_after_pruning_global_unstructured_l1 = correct_predictions / len(validation_set)
    print("Accuracy after L1 pruning:", accuracy_after_pruning_global_unstructured_l1)

    results_global_unstructured_l1[i] = accuracy_after_pruning_global_unstructured_l1

    # Reset the model to its original state (remove pruning)
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')

np.save('results_globalUnstructured_l1.npy', results_global_unstructured_l1)