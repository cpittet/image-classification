import torch
import torch.nn as nn


class Net(nn.Module):
    """
    CNN, takes 100x100 images with 3 channels as input
    """
    def __init__(self):
        super(Net, self).__init__()
        # "Verbose" definition to details everything...

        # Define all the functions that we want to learn the parameters of. Their parameters are automatically
        # registered as so, when setting the as attribute of the module

        # First convolution, stride of 3, kernels (7, 7) :
        # takes (100, 100) images with 3 channels as input
        # output (32, 32) with 96 channels
        conv1 = nn.Conv2d(3, 96, 7, stride=3)

        # input 96 channels (30, 30)
        # output 256 channels (30, 30)
        # kernel size 5, stride 1, padding 2
        conv2 = nn.Conv2d(96, 256, 5, padding=2)

        # input 256 channels (15, 15)
        # output 384 channels (15, 15)
        # kernel size 3, stride 1, padding 1
        conv3 = nn.Conv2d(256, 384, 3, padding=1)

        # input 384 channels (15, 15)
        # output 384 channels (15, 15)
        # kernel size 3, stride 1, padding 1
        conv4 = nn.Conv2d(384, 384, 3, padding=1)

        # input 384 channels (15, 15)
        # output 256 channels (15, 15)
        # kernel size 3, stride 1, padding 1
        conv5 = nn.Conv2d(384, 256, 3, padding=1)

        self.convs = nn.Sequential(
            conv1,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            conv2,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3,
            nn.ReLU(inplace=True),
            conv4,
            nn.ReLU(inplace=True),
            conv5,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # Fully connected layer 1 : in 256 * 7 * 7 = 12'544
        # output 4096
        fc1 = nn.Linear(256*7*7, 2048)

        # Fully connected layer 2 : in 4096
        # output 4096
        fc2 = nn.Linear(2048, 516)

        # Fully connected layer 3 : in 4096
        # output 120 (categories)
        fc3 = nn.Linear(516, 120)

        self.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            fc1,
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            fc2,
            nn.ReLU(inplace=True),
            fc3
        )

    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
