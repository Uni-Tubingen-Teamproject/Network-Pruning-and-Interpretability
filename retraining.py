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

# load the GoogleNet model
# model = models.googlenet(aux_logits=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', weights='googlenet', aux_logits=True)
model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', weights='GoogLeNet_Weights.DEFAULT', aux_logits=True)
model.train()  # Set the model to training mode


# Enable auxiliary heads for training if they exist
if hasattr(model, 'aux1'):
    model.aux1.training = True
if hasattr(model, 'aux2'):
    model.aux2.training = True


# # enable auxiliary heads for training
# model.aux1.training = True
# model.aux2.training = True

# move model to GPU if available
if torch.cuda.is_available():
    model = model.cuda()

# validation loop for the ffcv imagenet
validation_transformation = transforms.Compose([
    transforms.Resize(256),            
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_transformation = transforms.Compose([
    transforms.Resize(256),            
    transforms.CenterCrop(224),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(degrees=30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# # Load the ImageNet validation set
# validation_set = datasets.ImageFolder('/Users/szagu/OneDrive/Desktop/teamprojekt/imagenette2-320', transform=validation_transformation)

# validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=10, 
#                                                 sampler=torch.utils.data.SubsetRandomSampler(range(50)),
#                                                 num_workers=0)

# # Load the Imagenette training set
# train_set = datasets.ImageFolder('/Users/szagu/OneDrive/Desktop/teamprojekt/imagenette2-320', transform=train_transformation)

# # Create a DataLoader for the training subset
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=10, 
#                                            sampler=torch.utils.data.SubsetRandomSampler(range(50)), 
#                                            num_workers=0)


batch_size = 128

validation_set = datasets.ImageNet(root='/mnt/qb/datasets/ImageNet2012', split='val', transform=validation_transformation)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=8)

train_set = datasets.ImageNet(root='/mnt/qb/datasets/ImageNet2012', split='train', transform=train_transformation)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)

# function to validate the model
def validate(model, loader):
    model.eval()
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad(): ## during validation gradient is not computed to save storage space
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
    model.train()

    # Enable auxiliary heads for training if they exist
    if hasattr(model, 'aux1'):
        model.aux1.training = True
    if hasattr(model, 'aux2'):
        model.aux2.training = True

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


# function to calculate validation loss
# def validate_loss(model, criterion, loader):
#     model.eval()
#     total_loss = 0.0

#     with torch.no_grad():
#         for images, labels in loader:
#             if torch.cuda.is_available():
#                 images = images.to('cuda')
#                 labels = labels.to('cuda')

#             outputs = model(images)
#             if isinstance(outputs, tuple):
#                 outputs = outputs[0]

#             loss = criterion(outputs, labels)
#             total_loss += loss.item()

#     return total_loss / len(loader)

# calculate accuracy for the validation set before pruning
accuracy_before_pruning = validate(model, validation_loader)
print("Accuracy before pruning:", accuracy_before_pruning)

## -----------------------------------------------------------------------------------------------
## L1 UNSTRUCTURED PRUNING (GLOBAL)
##------------------------------------------------------------------------------------------------

initial_state = copy.deepcopy(model.state_dict())

# pruning amount for the loop
# amounts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
amounts = [0.3, 0.5, 0.7]

# initialize results dictionary
results_l1_unstructured_test = np.zeros(len(amounts))
epochs = 3

criterion = nn.CrossEntropyLoss()

initial_lr = 0.01  # Store the initial learning rate
optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=0.0001)
# optimizer = optim.SGD(model.parameters(), lr=0.1)

# scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

best_val_loss = float('inf')
patience = 5
patience_counter = 0

# apply l1 unstructured pruning to all layers with weights
parameters_to_prune = []
for name, module in model.named_modules():
    if hasattr(module, 'weight') and 'aux' not in name:
        parameters_to_prune.append((module, 'weight'))

# prune the model
for i, amt in enumerate(amounts):
    print(f"Pruning Rate: {amt}")

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amt,
    )

    accuracy_l1_pruned = validate(model, validation_loader)
    print("Accuracy after pruning:", accuracy_l1_pruned)

    # Reset the learning rate to the initial value before retraining
    for param_group in optimizer.param_groups:
        param_group['lr'] = initial_lr

    # Reset the scheduler if needed (if using a scheduler that depends on step count)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # retrain the model
    train(model, train_loader, criterion, optimizer, scheduler, epochs, validation_loader)

    # validate accuracy after pruning
    accuracy_l1_retrained = validate(model, validation_loader)
    print("Accuracy after retraining:", accuracy_l1_retrained)

    # save accuracy in the results array
    results_l1_unstructured_test[i] = accuracy_l1_retrained

    model.load_state_dict(initial_state)

    # note: do not remove pruning masks here to keep them for retraining

# save results
np.save('results_l1_unstructured_test.npy', results_l1_unstructured_test)

print("Finished pruning, retraining, and evaluation.")