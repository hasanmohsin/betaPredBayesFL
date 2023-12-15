import torch
import torch.nn as nn
import torch.nn.functional as F

####################################################
# IMPLEMENTATIONS FOR NETWORKS USED 
####################################################
class LinearNet(nn.Module):
    def __init__(self, inp_dim, num_hidden, out_dim):
        super().__init__()

        self.input_dim = inp_dim
        self.num_hidden = num_hidden
        self.out_dim = out_dim

        self.fc1 = nn.Linear(inp_dim, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, out_dim)
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze()
        return x

#for regression tasks: output a mean and a variance (to evaluate calibration)
# variance is assumed heteroscedatic ie. a function of input
class LinearNetVar(nn.Module):
    def __init__(self, inp_dim, num_hidden, out_dim):
        super().__init__()

        self.input_dim = inp_dim
        self.num_hidden = num_hidden
        self.out_dim = out_dim

        self.fc1 = nn.Linear(inp_dim, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, out_dim) # output mean 
        self.fc4 = nn.Linear(num_hidden, out_dim) #output var 
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = F.relu(self.fc1(x))
        layer = F.relu(self.fc2(x))
        x = self.fc3(layer).squeeze()
        var = self.fc4(layer).squeeze().exp() #so its positive
        return x, var

class CNN(nn.Module):
    def __init__(self, num_classes = 10):
        super(CNN, self).__init__()
        self.out_dim = num_classes

        self.conv1 = nn.Conv2d(3,   64,  3)
        self.conv2 = nn.Conv2d(64,  128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#for CIFAR10, and CIFAR100
class CNN_old(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        self.out_dim = num_classes

        self.conv1 = nn.Conv2d(in_channels = 3, 
                               out_channels = 6,
                              kernel_size = 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

