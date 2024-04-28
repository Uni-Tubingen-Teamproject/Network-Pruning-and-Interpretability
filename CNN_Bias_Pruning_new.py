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


### UNSTRUCTURED BIAS PRUNING PRUNING OF THE CNN 

## GETTING THE BIAS OF THE CNN-LAYERS
# convolutional layer 1
total_bias_count = 0

for name, param in module1.named_parameters():
    if 'bias' in name:
        total_bias_count += param.numel()  

print("Bias amount L1:", total_bias_count)
# Bias amount L1: 6


# convolutional layer 2
total_bias_count2 = 0

for name, param in module2.named_parameters():
    if 'bias' in name:
        total_bias_count2 += param.numel()  

print("Bias amount L2:", total_bias_count2)
# Bias amount L2: 16


# fully connected layer 1
bias_count1 = 0

for name, param in fc1.named_parameters():
    if 'bias' in name:
        bias_count1 += param.numel()  

print("Bias amount FCL1:", bias_count1)
# Bias amount FCL1: 120


# fully connected layer 2
bias_count2 = 0

for name, param in fc2.named_parameters():
    if 'bias' in name:
        bias_count2 += param.numel()  

print("Bias amount FCL2:", bias_count2)
# Bias amount FCL2: 84


# fully connected layer 3
bias_count3 = 0

for name, param in fc3.named_parameters():
    if 'bias' in name:
        bias_count3 += param.numel()  

print("Bias amount FCL3:", bias_count3)
# Bias amount FCL3: 10

## PRUNING
# prune randomly in a systematic fashion by increasing the amount of pruned weights by 0.1
# test random pruning for each amount 100 to calculate mean and sd
amounts_bias = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# can be set to an arbitrary value; runs per parameter pairing 
runs_bias = 2

# define the layers to be pruned
module1 = net.conv1
module2 = net.conv2
fc1 = net.fc1
fc2 = net.fc2
fc3 = net.fc3


results_bias = np.zeros((len(amounts_bias), len(amounts_bias), len(amounts_bias), len(amounts_bias), len(amounts_bias), runs_bias))

## UPDATED CODE FOR BIAS PRUNING OF ALL LAYERS
if os.path.exists('results_bias.npy') == False:
    for index_i, i in enumerate(amounts_bias):
        for index_j, j in enumerate(amounts_bias):
            for index_k, k in enumerate(amounts_bias):
                for index_l, l in enumerate(amounts_bias):
                    for index_m, m in enumerate(amounts_bias):

                        
                        if i == 0 and j == 0 and k == 0 and l == 0 and m == 0:
                            continue

                        
                        for n in range(runs_bias):

                            #reload the original model
                            net.load_state_dict(torch.load(PATH))

                            # pruning the bias of all layers
                            prune.random_unstructured(module1, name = "bias", amount = i)
                            prune.random_unstructured(module2, name = "bias", amount = j)
                            prune.random_unstructured(fc1, name = "bias", amount = k)
                            prune.random_unstructured(fc2, name = "bias", amount = l)
                            prune.random_unstructured(fc3, name = "bias", amount = m)
                            

                            #Assess the accuracy of the pruned network 
                            correct = 0
                            total = 0

                            #Test performance for this instance
                            with torch.no_grad():
                                for data in testloader:
                                    images, labels = data
                                    outputs = net(images)
                                    _, predicted = torch.max(outputs.data, 1)
                                    total += labels.size(0)
                                    correct += (predicted == labels).sum().item()

                            print(i, j, k, l, m, round(correct / total, 4)) # include k, l, m
                
                            results_bias[index_i][index_j][index_k][index_l][index_m][n] = round(correct / total, 4)

                            # reset to original state
                            prune.remove(module1, "bias")
                            prune.remove(module2, "bias")
                            prune.remove(fc1, "bias")
                            prune.remove(fc2, "bias")
                            prune.remove(fc3, "bias")
                            

    print('Finished Unstructured Pruning for Biases')

    #save results
    np.save('results_bias.npy', results_bias)
    loaded_results_bias = np.load('results_bias.npy')
else: 
    loaded_results_bias = np.load('results_bias.npy')
    print('Pruned model results loaded successfully!')


# Get the mean accuracy and the standard deviation 
mean_accuracy_bias = np.mean(loaded_results_bias, axis = 5)
std_accuracy_bias = np.std(loaded_results_bias, axis = 5)        

# Display results for bias pruning 
results_list_bias = []

for index_i, i in enumerate(amounts_bias):
    for index_j, j in enumerate(amounts_bias):
        for index_k, k in enumerate(amounts_bias):
            for index_l, l in enumerate(amounts_bias):
                for index_m, m in enumerate(amounts_bias):
                    if i == 0 and j == 0 and k == 0 and l == 0 and m == 0:
                        continue
                    
                    if index_i < mean_accuracy_bias.shape[0] and index_j < mean_accuracy_bias.shape[1] and index_k < mean_accuracy_bias.shape[2] and index_l < mean_accuracy_bias.shape[3] and index_m < mean_accuracy_bias.shape[4]:
                        row_bias = {
                            "RP Bias Conv1": i,
                            "RP Bias Conv2": j,
                            "RP Bias FC1": k,
                            "RP Bias FC2": l,
                            "RP Bias FC3": m,
                            "Mean Accuracy": mean_accuracy_bias[index_i, index_j, index_k, index_l, index_m],
                            "Standard Deviation": std_accuracy_bias[index_i, index_j, index_k, index_l, index_m]
                            }
                        
                    results_list_bias.append(row_bias)

results_df_bias = pd.DataFrame(results_list_bias)


pd.set_option('display.max_rows', None)

# Print the results for unstructured pruning of the bias
print('The following table contains the results of our systematic, repeated pruning and testing of the Bias of all layers:')
print(results_df_bias)
