import torch
from torch import nn
import torch.nn.functional as F

class Net_1(nn.Module):

    def __init__(self):
        super(Net_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 15, 7, padding=2)
        self.conv2 = nn.Conv1d(15, 20, 6, padding=2)
        self.conv3 = nn.Conv1d(20, 15, 5, padding=2)
        self.conv4 = nn.Conv1d(15, 10, 5, padding=2)
        self.conv5 = nn.Conv1d(10, 7, 4, padding=2)
        self.conv6 = nn.Conv1d(7, 7, 4, padding=2)
        self.conv7 = nn.Conv1d(7, 7, 4, padding=2)
        self.conv8 = nn.Conv1d(7, 7, 4, padding=2)
        self.conv9 = nn.Conv1d(7, 7, 4, padding=1)
        self.conv10 = nn.Conv1d(7, 1, 1)

    def forward(self, x):
        leaky_relu = nn.LeakyReLU(0.05)
        drop_out = nn.Dropout(0.9)
        x = torch.unsqueeze(x, 1)
        x = leaky_relu(self.conv1(x))
        x = leaky_relu(self.conv2(x))
        x = leaky_relu(self.conv3(x))
        x = leaky_relu(self.conv4(x))
        x = leaky_relu(self.conv5(x))
        x = leaky_relu(self.conv6(x))
        x = leaky_relu(self.conv7(x))
        x = leaky_relu(self.conv8(x))
        x = leaky_relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = x.squeeze(1)
        return x
