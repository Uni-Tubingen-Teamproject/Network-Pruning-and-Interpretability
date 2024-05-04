# all required imports
import torch
import torchvision.datasets as datasets
from torchvision import transforms
import json
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
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

# Define batch size, depending on available memory 
batch_size = 1

# Load the ImageNet validation set
validation_set = datasets.ImageNet(root='./data', split='val', transform=validation_transformation, download=True)

# Create a data loader for the validation set
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=0)

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

# Calculate the accuracy for the validation set before pruning
accuracy_before_pruning = correct_predictions / len(validation_set)
print("Accuracy before pruning:", accuracy_before_pruning)

## Global unstructured pruning for the whole model

# Pruning amount for the loop
amount_pruning = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Loop through different pruning rates
for pruning_rate in amount_pruning:
    print(f"Pruning Rate: {pruning_rate}")
    
    # Apply global unstructured pruning to the entire model

    # Collect all parameters in the model that can be pruned
    parameters_to_prune = []
    for name, module in model.named_modules():
            parameters_to_prune.append((module, 'weight'))

    # Prune the model
    prune.global_unstructured(
        parameters_to_prune,
        amount=pruning_rate,
    )

    # Count non-zero trainable parameters after pruning
    total_params = sum(p.numel() for p in model.parameters())
    non_zero_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pruned_params = total_params - non_zero_params

    print("Total parameters:", total_params)
    print("Non-zero parameters after pruning:", non_zero_params)
    print("Pruned parameters:", pruned_params)

   
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
accuracy_after_pruning = correct_predictions / len(validation_set)
print("Accuracy after pruning:", accuracy_after_pruning)

# Reset the model to its original state (remove pruning)
prune.remove(model, 'weight')