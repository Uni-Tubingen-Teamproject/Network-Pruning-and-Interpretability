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

from torchvision.datasets import ImageNet
import warnings
import os
from torchvision.datasets.utils import check_integrity, extract_archive, verify_str_arg
from torchvision.datasets.imagenet import load_meta_file, parse_devkit_archive, META_FILE, parse_val_archive, parse_train_archive

class HackyImageNet(ImageNet):

    def __init__(self, root: str, devkit_loc="/mnt/qb/datasets/ImageNet2012/", split: str = 'train', transform=None, download = False, **kwargs):
        if download is True:
            msg = ("The dataset is no longer publicly accessible. You need to "
                   "download the archives externally and place them in the root "
                   "directory.")
            raise RuntimeError(msg)
        elif download is False:
            msg = ("The use of the download flag is deprecated, since the dataset "
                   "is no longer publicly accessible.")
            warnings.warn(msg, RuntimeWarning)

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
        

# Load the GoogleNet model
model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', weights='GoogLeNet_Weights.DEFAULT')
model.eval()

# Move the model to GPU if CUDA is available
if torch.cuda.is_available():
    model = model.cuda()

## Validation loop for the ffcv imagenet 

# Define the transformations of the input images
validation_transformation = transforms.Compose([
    transforms.Resize(256),               
    transforms.CenterCrop(224),          
    transforms.ToTensor(),                
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

print("CUDA Available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())

# Define batch size, depending on available memory 
batch_size = 120

# Beispiel zur Verwendung:
root_path = "/scratch_local/datasets/ImageNet2012"

validation_set = HackyImageNet(root=root_path, split='val', transform=validation_transformation)
# Load the ImageNet validation set
# validation_set = datasets.ImageNet(root='/scratch_local/datasets/ImageNet2012', split='val', transform=validation_transformation)

# Create a data loader for the validation set
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=8)

# Load the ImageNet class labels
with open('imagenet_class_index.json') as f:
    class_idx = json.load(f)


# Get the number of images in the validation set
validation_set_size = len(validation_set)
print("Size of the ImageNet validation set:", validation_set_size)


# # Load the ImageNet class labels
# with open('imagenet_class_index.json') as f:
#     class_idx = json.load(f)

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

# Calculate the accuracy for the validation set before pruning
accuracy_before_pruning = correct_predictions / len(validation_set)
print("Accuracy before pruning:", accuracy_before_pruning)

## Global unstructured pruning for the whole model

# Pruning amount for the loop
amounts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# create a multidimensional array that can store all the values
results_global_unstructured_l1 = np.zeros(len(amounts))

parameters_to_prune = []
for name, module in model.named_modules():
    # Prüfen, ob das Modul das Attribut 'weight' hat
    if hasattr(module, 'weight') and module.weight is not None:
        parameters_to_prune.append((module, 'weight'))

# Loop through different pruning rates
for i, pruning_rate in enumerate(amounts):
    print(f"Pruning Rate: {pruning_rate}")
    
    # Prune the model
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_rate,
    )

    # Count non-zero trainable parameters after pruning
    total_params = sum(p.numel() for p in model.parameters())
    non_zero_params = sum(p.numel() for p in model.parameters() if p.data.count_nonzero() > 0)
    pruned_params = total_params - non_zero_params

    print("Total parameters:", total_params)
    print("Non-zero parameters after pruning:", non_zero_params)
    print("Pruned parameters:", pruned_params)

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

