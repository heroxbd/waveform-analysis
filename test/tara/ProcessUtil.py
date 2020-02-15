import numpy as np
import torch
import torch.utils.data as Data
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import tables

#Neural Networks
class Net_1(nn.Module):

    def __init__(self):
        super(Net_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 25, 9, padding=4)
        self.conv2 = nn.Conv1d(25, 16, 7, padding=3)
        self.conv3 = nn.Conv1d(16, 10, 7, padding=3)
        self.conv4 = nn.Conv1d(10, 1, 1)

    def forward(self, x):
        leaky_relu = nn.LeakyReLU(0.05)
        drop_out = nn.Dropout(0.9)
        x = torch.unsqueeze(x, 1)
        x = leaky_relu(self.conv1(x))
        x = leaky_relu(self.conv2(x))
        x = leaky_relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.squeeze(1)
        return x
    
# Define the database columns
class AnswerData(tables.IsDescription):
    EventID    = tables.Int64Col(pos=0)
    ChannelID  = tables.Int16Col(pos=1)
    PETime     = tables.Int16Col(pos=2)
    Weight     = tables.Float32Col(pos=3)