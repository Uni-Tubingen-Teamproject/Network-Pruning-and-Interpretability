## implementation CNN with fashionMNIST dataset

# all required packages and imports; pip install them
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd  # type: ignore
import os

# some fix
torch.multiprocessing.set_start_method('spawn')

# normalisation of dataset 
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])
# shouldn't this be (0,), (1,)? Have not tried it out, was just what I know from my
# ML lecture (Florian)

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

# classes form the FashionMNIST dataset 
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


# save training progress if there is no save of the model's parameters
PATH = "./CNN_pruned.pth"

if not os.path.exists(PATH):
    # start training of the network and save the progress if no training progress was saved
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

    print('Finished Training!')

    torch.save(net.state_dict(), PATH)  

    print('Model parameters saved successfully!')

# load the model's parameters from where we stored them
if os.path.exists(PATH):
    net.load_state_dict(torch.load(PATH))
    print('Model parameters loaded successfully!')
else:
    print(f"No model parameters found at '{PATH}'.")


# validate model to obtain the model's mean accuracy 
# new testloader to enable shuffling of testdata
testloader_shuffle = torch.utils.data.DataLoader(trainset, batch_size = batch_size, 
                                                 shuffle = True, num_workers = 0)

# determines how many runs we do, arbitrarily set to 3
n_simulations = 3

# vector to save the accuracys from the simulations
accuracy_model_unpruned = [] 

if os.path.exists('mean_accuracy.npy') == False:

    for i in range(n_simulations): 

        # reset the parameters each time
        correct = 0
        total = 0

        with torch.no_grad():
            for data in testloader_shuffle:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # save the obtained accuracy as a float with two decimals (so here 4)
        accuracy_model_unpruned.append(round(correct / total, 4))

    # calculate the mean accuracy of the unpruned model over n_simulations testing runs 
    mean_accuracy_model_unpruned = np.sum(accuracy_model_unpruned) / len(accuracy_model_unpruned)

    # Save it
    np.save('mean_accuracy.npy', mean_accuracy_model_unpruned)

mean_accuracy_model_unpruned_loaded = np.load('mean_accuracy.npy')
print('The mean accuracy of the unpruned model is:', mean_accuracy_model_unpruned_loaded)


### UNSTRUCTURED PRUNING OF THE CNN 


## 1) UNSTRUCTURED PRUNING OF CONVOLUTION LAYERS

# prune randomly in a systematic fashion by increasing the amount of pruned weights by 0.1
# test random pruning for each amount 100 to calculate mean and sd
amounts = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# can be set to an arbitrary value; runs per parameter pairing 
runs = 2

# define the layers to be pruned; we do this first for the convolutional layers
module1 = net.conv1
module2 = net.conv2

# print parameters if wanted
# print(list(module.named_parameters()))
# print(list(module2.named_parameters()))


# create a multidimensional array that can store all the values
results = np.zeros((len(amounts), len(amounts), runs))

# if the unstructured pruning has not been conducted, do it
# else, load the testing results for the pruned network

if os.path.exists('results.npy') == False:
    for index_i, i in enumerate(amounts):
        for index_j, j in enumerate(amounts):

            # jump instances where both pruning parameters are 0
            if i == 0 and j == 0:
                    continue

            # to obtain the mean, run the testing 100-times
            for k in range(runs):

                # reload the original model
                net.load_state_dict(torch.load(PATH))

                # Instantiate the pruning in this loop to obtain newly random pruned net each time
                prune.random_unstructured(module1, name = "weight", amount = i)
                prune.random_unstructured(module2, name = "weight", amount = j)

                # Assess the accuracy of the pruned network 
                correct = 0
                total = 0

                # Test performance for this instance
                with torch.no_grad():
                    for data in testloader:
                        images, labels = data
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                print(i, j, round(correct / total, 4))
                
                results[index_i][index_j][k] = round(correct / total, 4)

                # reset the net to its original state
                prune.remove(module1, "weight")
                prune.remove(module2, "weight")

    print('Finished Unstructured Pruning')

    # save the results
    np.save('results.npy', results)
    loaded_results = np.load('results.npy')
else: 
    loaded_results = np.load('results.npy')
    print('Pruned model results loaded successfully!')

# we can print them if wanted
# print(loaded_results)

# Get the mean accuracy and the standard deviation 
mean_accuracy = np.mean(loaded_results, axis = 2)
std_accuracy = np.std(loaded_results, axis = 2)        

# Display the results with a dataframe
results_list = []

for index_i, i in enumerate(amounts):
    for index_j, j in enumerate(amounts):

        if i == 0 and j == 0:
            continue

        if index_i < mean_accuracy.shape[0] and index_j < mean_accuracy.shape[1]:
            row = {
                "RP Conv1": i,
                "RP Conv2": j,
                "Mean Accuracy": mean_accuracy[index_i, index_j],
                "Standard Deviation": std_accuracy[index_i, index_j]
            }
        results_list.append(row)

results_df = pd.DataFrame(results_list)

# to view all rows, we need to enable the following options
pd.set_option('display.max_rows', None)

# Print the results for unstructured pruning
print('The following table contains the results of our systematic, repeated pruning and testing:')
print(results_df)

## 2) UNSTRUCTURED PRUNING OF FULLY CONNECTED LAYERS

# Define the fully connected layers for pruning
module3 = net.fc1
module4 = net.fc2
module5 = net.fc3

# define the amounts for pruning
amounts_fc = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# create a multidimensional array to store the results
results_fc = np.zeros((len(amounts_fc), len(amounts_fc), len(amounts_fc), runs))  # Include third dimension for fc3

# if unstructured pruning has not been conducted, do it
# otherwise, load the testing results for the pruned network

if not os.path.exists('results_fc.npy'):
    for index_i, i in enumerate(amounts_fc):
        for index_j, j in enumerate(amounts_fc):
            for index_k, k in enumerate(amounts_fc):  # Include loop for third fully connected layer

                # skip instances where all pruning parameters are 0
                if i == 0 and j == 0 and k == 0:
                    continue

                # perform testing multiple times to obtain mean accuracy
                for l in range(runs):

                    # reload the original model
                    net.load_state_dict(torch.load(PATH))

                    # instantiate pruning in this loop to obtain newly random pruned net each time
                    prune.random_unstructured(module3, name="weight", amount=i)
                    prune.random_unstructured(module4, name="weight", amount=j)
                    prune.random_unstructured(module5, name="weight", amount=k)  # Include pruning for fc3

                    # assess accuracy of pruned network 
                    correct = 0
                    total = 0

                    # test performance for this instance
                    with torch.no_grad():
                        for data in testloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()

                    print(i, j, k, round(correct / total, 4))  # Include k in the print statement

                    results_fc[index_i][index_j][index_k][l] = round(correct / total, 4)

                    # reset the net to its original state
                    prune.remove(module3, "weight")
                    prune.remove(module4, "weight")
                    prune.remove(module5, "weight")  # Remove pruning for fc3

    print('Finished Unstructured Pruning of Fully Connected Layers')

    # save the results
    np.save('results_fc.npy', results_fc)
else: 
    loaded_results_fc = np.load('results_fc.npy')
    print('Pruned model results loaded successfully!')

# calculate mean accuracy and standard deviation
mean_accuracy_fc = np.mean(loaded_results_fc, axis=(2, 3))
std_accuracy_fc = np.std(loaded_results_fc, axis=(2, 3))

# display results with a dataframe
results_list_fc = []

for index_i, i in enumerate(amounts_fc):
    for index_j, j in enumerate(amounts_fc):
        for index_k, k in enumerate(amounts_fc):

            if i == 0 and j == 0 and k == 0:
                continue

            if index_i < mean_accuracy_fc.shape[0] and index_j < mean_accuracy_fc.shape[1] and index_k < mean_accuracy_fc.shape[2]:
                row_fc = {
                    "RP FC1": i,
                    "RP FC2": j,
                    "RP FC3": k,  # Include RP FC3
                    "Mean Accuracy": mean_accuracy_fc[index_i, index_j, index_k],
                    "Standard Deviation": std_accuracy_fc[index_i, index_j, index_k]
                }
                results_list_fc.append(row_fc)

results_df_fc = pd.DataFrame(results_list_fc)

# print results for unstructured pruning of fully connected layers
print('The following table contains the results of systematic, repeated pruning and testing of Fully Connected Layers:')
print(results_df_fc)