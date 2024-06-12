import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision import models

from ffcv_dataloaders import create_train_loader, create_test_loader

FFCV_PATH = "/mnt/lustre/datasets/ffcv_imagenet_data"

# Check if CUDA is available and set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create data loaders
train_loader = create_train_loader(
    os.path.join(FFCV_PATH, 'train_500_0.50_90.ffcv'),
    num_workers=8,
    batch_size=64,
    distributed=False,
    in_memory=False,
    device=device
)

val_loader = create_test_loader(
    os.path.join(FFCV_PATH, 'val_500_0.50_90.ffcv'),
    num_workers=8,
    batch_size=64,
    distributed=False,
    in_memory=False,
    device=device
)

scaler = GradScaler()

# Initialize the ResNet50 model
model = models.googlenet()
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=1e-4)

# Training and validation loops
num_epochs = 25

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / total
    train_acc = 100. * correct / total

    # Validation phase
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            with autocast():
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / total
    val_acc = 100. * correct / total

    print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% '
          f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%')

print('Training complete')
