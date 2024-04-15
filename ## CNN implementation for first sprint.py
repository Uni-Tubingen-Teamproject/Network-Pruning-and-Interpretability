## implementation CNN with fashionMNIST dataset

# all required imports
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

# Some crazy fix
torch.multiprocessing.set_start_method('spawn')

# normalisation of dataset 
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])
# shouldn't this be (0,), (1,)? Have not tried it out, was just what I know from my
# ML lecture

# how many images are processed until the model reevaluates its parameters
batch_size = 4

# load training set 
trainset = torchvision.datasets.FashionMNIST(root = './data', train = True, 
                                             download = True, transform = transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, 
                                          shuffle = True, num_workers = 0)

# load test/validation set 
testset = torchvision.datasets.FashionMNIST(root = './data', train = False, 
                                             download = True, transform = transform)

testloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, 
                                          shuffle = False, num_workers = 0)

# classes form the fashionMNIST dataset 
classes = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt",
           "Sneaker", "Bag", "Ankle boot")


# Define the CNN 
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # 1 = grey tone (no rgb), 6 = output channel, 5 = kernel size
        self.pool = nn.MaxPool2d(2, 2) # 2 = kernel size, 2 = stride
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()


# define a loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

# start training 
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')


# save training progress 
PATH = "./CNN_implementation.py"
torch.save(net.state_dict(), PATH)

# validate model
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# My last attempt resulted in a loss of 0.344 and an accuracy of 87% >>> 10% chance
# One weird thing happens after the code compiled; everything turned into red-coloured
# gibberish... Idk why. Ok, it didn't happen in this version here on Git, I don't know what
# what is going on...