## implementation GoogleNet

# all required imports
import torch
import urllib
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import json
import torchvision.datasets as datasets



# Load the GoogleNet model
model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', weights='GoogLeNet_Weights.DEFAULT')
model.eval()

## Validation loop for the ffcv imagenet

validation_transformation = transforms.Compose([
    transforms.Resize(256),            
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define batch size, depending on available memory
batch_size = 120

validation_set = datasets.ImageNet(root='/mnt/qb/datasets/ImageNet2012', split='val', transform=validation_transformation)

# Create a data loader for the validation set
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=8)

if torch.cuda.is_available():
    model = model.cuda()

### MAKE A SUBSET OF 100 PICTURES
# validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=1, 
#                                                 sampler=torch.utils.data.SubsetRandomSampler(range(50)),
#                                                 num_workers=0)


# Load the ImageNet class labels
# extract class labels for imagenette
# classes = validation_set.classes


# with open('imagenet_class_index.json') as f:
#     class_idx = json.load(f)

#classes = {idx: label for idx, label in class_idx.items()}

# Counters to compute accuracy
correct_predictions = 0


print("Hello World!")

for images, labels in validation_loader:

    # Move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        images = images.to('cuda')
        labels = labels.to('cuda')

    # Execute the model
    with torch.no_grad():
        output = model(images)

    # Get the prediction for each image
    _, prediction = torch.max(output, 1)

    # Calculate the number of correctly predicted images from the validation set
    correct_predictions += (prediction == labels).sum().item()

# Calculate the accuracy for the validation set before pruning
accuracy_before_pruning = (correct_predictions / len(validation_set)) * 100
print("Accuracy before pruning:", accuracy_before_pruning)

## -----------------------------------------------------------------------------------------------
## L1 UNSTRUCTERED PRUNING
##------------------------------------------------------------------------------------------------

# pruning amount for the loop
amounts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# count number of modules with weights
count = 0

for name, module in model.named_modules():
    if hasattr(module, "weight"):
        count += 1


# initialize the results-dictionary
results_l1 = {}

# prune the model
for name, module in model.named_modules():

    if hasattr(module, "weight"):
        results_l1[name] = {} # initialize empty dictionary for each module

        for amt in amounts:
            print(f"Pruning Rate: {amt}, Module: {name}")

            # prune the amount specified above
            prune.l1_unstructured(module, 'weight', amount=amt)

            # Validate accuracy after pruning
            correct_predictions = 0

            # Loop through the validation set and compute accuracy after pruning
            for images, labels in validation_loader:
                if torch.cuda.is_available():
                    images = images.to('cuda')
                    labels = labels.to('cuda')

                with torch.no_grad():
                    output = model(images)

                _, prediction = torch.max(output, 1)
                correct_predictions += (prediction == labels).sum().item()

            # Calculate the accuracy for the validation set after pruning
            accuracy_l1 = (correct_predictions / len(validation_set)) * 100
            print("Accuracy after pruning:", accuracy_l1)

            # Save accuracy in your dictionary
            results_l1[name][amt] = accuracy_l1

            # set model to original state
            prune.remove(module, 'weight')



# save results
np.save('results_l1.npy', results_l1)