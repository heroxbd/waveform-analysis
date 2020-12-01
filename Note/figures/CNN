import torch
from torch import nn
import torch.nn.functional as F

class Net_1(nn.Module):

    def __init__(self):
        super(Net_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 25, 21, padding=10)
        self.conv2 = nn.Conv1d(25, 20, 17, padding=8)
        self.conv3 = nn.Conv1d(20, 15, 13, padding=6)
        self.conv4 = nn.Conv1d(15, 10, 9, padding=4)
        self.conv5 = nn.Conv1d(10, 1, 1)

    def forward(self, x):
        leaky_relu = nn.LeakyReLU(0.05)
        drop_out = nn.Dropout(0.9)
        x = torch.unsqueeze(x, 1)
        x = leaky_relu(self.conv1(x))
        x = leaky_relu(self.conv2(x))
        x = leaky_relu(self.conv3(x))
        x = leaky_relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.squeeze(1)
        return x

