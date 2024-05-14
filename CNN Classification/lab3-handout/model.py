import torch
from torch import nn


class YourFirstNet(torch.nn.Module):
    def __init__(self, n_labels):
        super(YourFirstNet, self).__init__()

        # Conv-block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Conv-block
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

       # Linear-Block
        self.fc1 = nn.Linear(in_features=54450, out_features=33)
        self.relu3 = nn.ReLU()

        # Linear-Block
        self.fc2 = nn.Linear(in_features=33, out_features=n_labels)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.maxpool1(self.relu1(self.conv1(x)))

        x = self.maxpool2(self.relu2(self.conv2(x)))

        x = torch.flatten(x, start_dim=1)
        x = self.relu3(self.fc1(x))

        x = self.fc2(x)

        output = self.logSoftmax(x)
        return output
    
class YourSecondNet(torch.nn.Module):
    def __init__(self, n_labels):
        super(YourSecondNet, self).__init__()

        # Conv-block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=50, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Conv-block
        self.conv2 = nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv3 = nn.Conv2d(in_channels=100, out_channels=75, kernel_size=(5, 5))
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

       # Linear-Block
        self.fc1 = nn.Linear(in_features=14700, out_features=500)
        self.relu = nn.ReLU()

        # Linear-Block
        self.fc2 = nn.Linear(in_features=500, out_features=n_labels)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.maxpool1(self.relu1(self.conv1(x)))

        x = self.maxpool2(self.relu2(self.conv2(x)))

        x = self.maxpool3(self.relu3(self.conv3(x)))

        # print(x.shape)
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))

        x = self.fc2(x)

        output = self.logSoftmax(x)
        return output