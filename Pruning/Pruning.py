## implementation CNN with fashionMNIST dataset

# all required imports
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torch.optim as optim

# neccesaary fix
torch.multiprocessing.set_start_method('spawn')

# normalisation of dataset 
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])   # normalize the tensor with mean (0.5) and standard deviation (0.5)


# how many images are processed until the model reevaluates its parameters
# try out different values and look at changes in loss and accuracy: 
# for batch_size = 16: loss = 0.428, accuracy = 83%
# for batch_size = 8: loss = 0.428, accuracy = 85%
# for batch_size = 4: loss = 0.344, accuracy = 87%
# for batch_size = 2: loss = 0.345, accuracy = 88%
# for batch_size = 1: loss = 0.380, accuracy = 87%
batch_size = 4

# load training set 
trainset = torchvision.datasets.FashionMNIST(root = './data', train = True, 
                                             download = True, transform = transform)
# create a DataLoader for the training set
trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, 
                                          shuffle = True, num_workers = 0)

# load test set 
testset = torchvision.datasets.FashionMNIST(root = './data', train = False, 
                                             download = True, transform = transform)
# create a DataLoader for the training set
testloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, 
                                          shuffle = False, num_workers = 0)

# define the classes of the FashionMNIST dataset 
classes = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt",
           "Sneaker", "Bag", "Ankle boot")


# define the CNN 
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # first convolutional layer: input channels = 1 (grayscale), output channels = 6, kernel size = 5x5
        self.conv1 = nn.Conv2d(1, 6, 5) 
        self.pool = nn.MaxPool2d(2, 2) # kernel size = 2x2 and stride = 2
        # second convolutional layer: input channels = 6, output channels = 16, kernel size = 5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()


# define a loss function
criterion = nn.CrossEntropyLoss()

# Stochastic Gradient Descent (SGD) optimizer with learning rate lr = 0.001 and momentum = 0.9
# Learning rate (lr): determines the step size at which the optimizer adjusts the parameters during training
# Momentum: helps to smooth out these adjustments and navigate through the training process more efficiently
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

# # start training 
# for epoch in range(2):  # loop over the dataset multiple times

#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
#             running_loss = 0.0

# print('Finished Training')


# save training progress and define the file path where the trained model is saved
PATH = "./CNN_implementation.py"
#torch.save(net.state_dict(), PATH)

# load the trained model
net.load_state_dict(torch.load(PATH))

# validate model
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:     # iterate over the test dataset
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)                             # update total count of samples 
        correct += (predicted == labels).sum().item()       # update count of correctly predicted samples


# print the accuracy of the network on the test dataset
print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


# Random Pruning

# get the parameters of the first convolutional layer
module = net.conv1
# print the parameters of the first convolutional layer
print(list(module.named_parameters()))

# get the parameters of the second convolutional layer
module2 = net.conv2
# print the parameters of the second convolutional layer
print(list(module2.named_parameters()))

# prune 30% of the weights in the first convolutional layer randomly -> approx 87% accuracy
# prune 50% of the weights in the first convolutional layer randomly -> approx 68% accuracy
prune.random_unstructured(module, name="weight", amount=0.3)

# now we'll also prune the second layer
# prune 30% of the weights in the second layer and 30% in the first layer randomly -> approx 83% accuracy
# prune 50% of the weights in the second layer and 50% in the first layer randomly -> approx 63% accuracy
# prune 50% of the weights in the second layer and 30% in the first layer randomly -> approx 84% accuracy
# prune 30% of the weights in the second layer and 50% in the first layer randomly -> approx 61% accuracy
# only prune 30% of the weights in second layer randomly -> approx 87% accuracy
# only prune 50% of the weights in second layer randomly -> approx 84% accuracy
prune.random_unstructured(module2, name="weight", amount=0.3)


# assessing the accuracy after pruning
with torch.no_grad():           # disable gradient calculations while evaluating the pruned network
    for data in testloader:     # iterate over the test dataset
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)                              # update total count of samples
        correct += (predicted == labels).sum().item()        # update count of correctly predicted samples


# print the accuracy of the pruned network on the test dataset
print(f'Accuracy of the pruned network on the 10000 test images: {100 * correct // total} %')