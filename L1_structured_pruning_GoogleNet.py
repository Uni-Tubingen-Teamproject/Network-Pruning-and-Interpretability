## implementation GoogleNet

# all required imports
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


## Validation loop for the ffcv imagenet 

# Define the transformations of the input images
validation_transformation = transforms.Compose([
    transforms.Resize(256),               
    transforms.CenterCrop(224),          
    transforms.ToTensor(),                
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])  
])

# Define batch size, depending on available memory 
batch_size = 1

# Load the ImageNet validation set
validation_set = datasets.ImageNet(root = './data', split = 'val', transform = validation_transformation, download = True)

# Create a data loader for the validation set
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size = batch_size, shuffle = False, num_workers = 0)

# Load the ImageNet class labels
with open('imagenet_class_index.json') as f:
    class_idx = json.load(f)

classes = {idx: label for idx, label in class_idx.items()}

# Counters to compute accuracy
correct_predictions = 0

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


# Calculate the accuracy for the validation set 
accuracy = correct_predictions / len(validation_set)


## L1 Structured Pruning 

# Percentage of pruned parameters
amounts = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# count the number of modules that can be pruned
count = 0

for module in model.named_modules():
    if hasattr(module, 'weight'):
        count += 1

results_l1_structured = np.zeros(count, len(amounts))

for name, module in model.named_modules():
    module_index = 0
    if hasattr(module, 'weight'):
        for amt in amounts:
            prune.ln_structured(module, name = 'weight', amount = amt, n = 1, dim = 0)

            # Assess the accuracy and store it 
            correct_predictions = 0

            for images, labels in validation_loader:
                if torch.cuda.is_available():
                    images = images.to('cuda')
                    labels = labels.to('cuda')

                with torch.no_grad():
                    output = model(images)

                _, prediction = torch.max(output, 1)
                correct_predictions += (prediction == labels).sum().item()

            # Calculate the accuracy for the validation set after pruning
            accuracy_l1_structured = correct_predictions / len(validation_set)
            results_l1_structured[module_index][amt] = accuracy_l1_structured

            # Remove the pruning 
            prune.remove(module, 'weight')

        module_index += 1

# save the results
np.save('results_l1_structured.npy', results_l1_structured)