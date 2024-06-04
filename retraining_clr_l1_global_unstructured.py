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

class HackyImageNet(ImageNet):

    def init(self, root: str, devkit_loc="/mnt/qb/datasets/ImageNet2012/", split: str = 'train', transform=None, download=False, **kwargs):
        if download:
            raise NotImplementedError("Automatic download of the ImageNet dataset is not supported.")
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

# load the GoogleNet model
model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', weights='GoogLeNet_Weights.DEFAULT', aux_logits=True)
model.train()  # Set the model to training mode

# Enable auxiliary heads for training if they exist
if hasattr(model, 'aux1'):
    model.aux1.training = True
if hasattr(model, 'aux2'):
    model.aux2.training = True

# move model to GPU if available
if torch.cuda.is_available():
    model = model.cuda()

# exclude aux layers
exclude_layers = ['aux1.conv.conv', 'aux2.conv.conv']

# define which parameters to prune
parameters_to_prune = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d) and name not in exclude_layers:
        parameters_to_prune.append((module, 'weight'))

# define transforms for the validation set
validation_transformation = transforms.Compose([
    transforms.Resize(256),            
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# define transforms for the training set
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

# define batch size
root_path = "/scratch_local/datasets/ImageNet2012"

batch_size = 128

validation_set = HackyImageNet(root=root_path, split='val', transform=validation_transformation)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=8)

train_set = HackyImageNet(root=root_path, split='train', transform=train_transformation)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)

# function to validate the model
def validate(model, loader, criterion): 
    model.eval()
    correct_predictions = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            if torch.cuda.is_available():
                images = images.to('cuda')
                labels = labels.to('cuda')

            outputs = model(images)
            if isinstance(outputs, tuple):  # check if model has auxiliary outputs
                outputs = outputs[0]

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    # compute accuracy and average loss
    accuracy = (correct_predictions / total_samples) * 100
    avg_loss = total_loss / len(loader)
   
    # print validation results
    print(f'Validation Loss: {avg_loss}')
    return accuracy, avg_loss

# function to train the model
def train(model, loader, criterion, optimizer, scheduler, epochs=1):
    # set model to training mode
    model.train()

    # enable auxiliary heads for training if they exist
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

            # set all gradients to zero
            optimizer.zero_grad()

            outputs = model(images)

            if isinstance(outputs, tuple): 
                outputs, aux1, aux2 = outputs  
                loss1 = criterion(outputs, labels)
                loss2 = criterion(aux1, labels)
                loss3 = criterion(aux2, labels)

                loss = loss1 + 0.3 * loss2 + 0.3 * loss3
            else:
                loss = criterion(outputs, labels)
            
            loss.backward()
            # update model parameters based on computed gradients
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        # get current learning rate
        current_lr = scheduler.get_last_lr()[0]
        
        # Calculate and print training loss
        epoch_loss = running_loss / len(loader)
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {epoch_loss}, Learning Rate: {current_lr}')

# function to prune and retrain the model
def prune_and_retrain_sgd_clr(model, train_loader, validation_loader, amounts, criterion, base_lr, max_lr, step_size_up, epochs):
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    scheduler =  lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up, mode='triangular')

    results_pr_sgd_clr = np.zeros(len(amounts))

    initial_state = copy.deepcopy(model.state_dict())

    for i, amt in enumerate(amounts):
        print(f'Pruning Rate: {amt}')

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amt,
        )

        # validate pruned model
        accuracy_l1_pruned, _ = validate(model, validation_loader, criterion)
        print(f'Accuracy after pruning: {accuracy_l1_pruned} %')

        # Learning rate rewinding: reset learning rate to initial state
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr

        # Reset scheduler if needed
        scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up, mode='triangular')

        # Retrain model
        train(model, train_loader, criterion, optimizer, scheduler, epochs)

        # Compute accuracy and validation loss after pruning
        accuracy_l1_retrained, _ = validate(model, validation_loader, criterion)
        print(f'Accuracy after retraining with SGD and Exponential LR: {accuracy_l1_retrained} %')

        # Save accuracy
        results_pr_sgd_clr[i] = accuracy_l1_retrained

        for module, _ in parameters_to_prune:
            prune.remove(module, 'weight')
        
        model.load_state_dict(initial_state)

    return results_pr_sgd_clr

# specify criterion
criterion = nn.CrossEntropyLoss()

# calculate accuracy for the validation set before pruning
accuracy_before_pruning, _ = validate(model, validation_loader, criterion)
print(f'Accuracy before pruning: {accuracy_before_pruning} %')

# specify amounts
amounts = [0.3, 0.5, 0.7]

# call the prune and retrain function
results_pr_sgd_clr = prune_and_retrain_sgd_clr(model, train_loader, validation_loader, amounts, criterion, base_lr=0.001, max_lr=0.006, step_size_up=782, epochs=4)
np.save('results_pr_sgd_clr.npy', results_pr_sgd_clr)

print('Finished pruning, retraining, and evaluation.')
