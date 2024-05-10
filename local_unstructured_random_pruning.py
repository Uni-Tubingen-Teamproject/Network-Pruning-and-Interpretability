import torch
import torchvision.datasets as datasets
import urllib
from PIL import Image
from torchvision import transforms
import numpy as np
import torch.nn.utils.prune as prune
from torch.utils.data.sampler import SubsetRandomSampler

# Load the GoogleNet model
model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', weights='GoogLeNet_Weights.DEFAULT')
model.eval()

# Move the model to GPU if CUDA is available
if torch.cuda.is_available():
    model = model.cuda()

validation_transformation = transforms.Compose([
    transforms.Resize(256),               
    transforms.CenterCrop(224),          
    transforms.ToTensor(),                
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

# Pruning parameters and number of runs
amounts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
runs = 2 # number of runs per pruning amount

batch_size = 120

# create validation_loader
validation_set = datasets.ImageNet(root='/mnt/qb/datasets/ImageNet2012', split='val', transform=validation_transformation)

validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=8)

# Array for saving Accuracy for each pruning amount
accuracy_results = np.zeros((len(amounts), runs))

# Prune the model
for module_name, module in model.named_modules():
    if hasattr(module, 'weight'):
        for i, amt in enumerate(amounts):
            # Loop over the number of runs
            for run in range(runs):
                # Apply pruning to the module
                prune.random_unstructured(module, name='weight', amount=amt)

                # Validation loop after pruning
                correct_predictions = 0
                total_images = 0
                for images, labels in validation_loader:
                    # Move data to GPU if available
                    if torch.cuda.is_available():
                        images = images.to('cuda')
                        labels = labels.to('cuda')
                    
                    # Execute the model
                    with torch.no_grad():
                        output = model(images.to('cuda') if torch.cuda.is_available() else images)
                    
                    # Get the prediction for each image
                    _, prediction = torch.max(output, 1)
                    correct_predictions += (prediction == labels).sum().item()
                    total_images += labels.size(0)
                
                # Calculate the accuracy for the validation set after pruning
                accuracy_after_pruning = correct_predictions / total_images
                
                # save accuracy in array
                accuracy_results[i, run] = accuracy_after_pruning

                print(f"Run {run+1} - Accuracy after pruning with amount {amt}: {accuracy_after_pruning}")

                # Remove pruning from the layer
                prune.remove(module, 'weight') 

                # Check if the pruning amount is 0.9, if so, break out of the loop
                if amt == 0.9:
                    break

            # Check if the pruning amount is 0.9, if so, break out of the loop
            if amt == 0.9:
                break

        # Check if the pruning amount is 0.9, if so, break out of the loop
        if amt == 0.9:
            break

# Output the average accuracies for each pruning amount
for i, amt in enumerate(amounts):
    mean_accuracy = np.mean(accuracy_results[i])
    std_deviation = np.std(accuracy_results[i])
    print(f"Pruning amount: {amt}, Mean accuracy: {mean_accuracy}, Std deviation: {std_deviation}")

# save pruning results
np.save('pruning_results.npy', accuracy_results)

















