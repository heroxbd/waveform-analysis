import numpy as np
import re

import torch
import torch.utils.data as Data

from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

import os,sys
import time

import tables

# Make Saving_Directory
NetDir = sys.argv[1]
LoadPath= sys.argv[2]
SavePath = sys.argv[3]
if not os.path.exists(SavePath):
    os.makedirs(SavePath)

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

fileSet = os.listdir(NetDir)
matchrule = re.compile(r"_epoch(\d+)_loss(\d+(\.\d*)?|\.\d+)([eE]([-+]?\d+))?")
NetLoss_reciprocal = []
for filename in fileSet :
    if "_epoch" in filename : NetLoss_reciprocal.append(1/float(matchrule.match(filename)[2]))
    else : NetLoss_reciprocal.append(0)
net_name = fileSet[NetLoss_reciprocal.index(max(NetLoss_reciprocal))]
net = torch.load(NetDir+net_name).cuda(device=2)# Pre-trained Model Parameters

# Data Settings
LoadingPeriod= 3200
BATCHSIZE=3200
# h5 file handling
# Define the database columns
class AnswerData(tables.IsDescription):
    EventID    = tables.Int64Col(pos=0)
    ChannelID  = tables.Int16Col(pos=1)
    PETime     = tables.Int16Col(pos=2)
    Weight     = tables.Float32Col(pos=3)

# Create the output file and the group
# h5file = tables.open_file("./Prediction_Results/Prediction_Mod_ztraining.h5", mode="w", title="OneTonDetector")
h5file = tables.open_file(SavePath+"Prediction.h5", mode="w", title="OneTonDetector")


# Create tables
AnswerTable = h5file.create_table("/", "Answer", AnswerData, "Answer")
answer = AnswerTable.row


start_time = time.time()

# Loading Data
PreFile =  tables.open_file(LoadPath+"Pre.h5")
Data_set = PreFile.root.TestDataTable
WindowSize = len(Data_set[0]['Waveform'])
Total_entries = len(Data_set)
print(Total_entries)

entryList = np.arange(0,Total_entries,LoadingPeriod)
entryList = np.append(entryList,Total_entries)
#for k,entry in enumerate(entryList[0:-2]) :
for k,entry in enumerate(entryList[0:1]) :
    EventData = Data_set[entry:entryList[k+1]]['EventID']
    ChanData = Data_set[entry:entryList[k+1]]['ChannelID']
    WaveData = Data_set[entry:entryList[k+1]]['Waveform']
    # Making Dataset
    predict_data = torch.from_numpy(WaveData).cuda(device=2)
    predict_loader = Data.DataLoader(dataset=predict_data,batch_size=BATCHSIZE,shuffle=False)

    # Makeing Output
    Output_Data = []
    for i,data in enumerate(predict_loader,0):
        inputs = Variable(data).float().cuda(device=2)
        outputs = net(inputs)
        batch_output = outputs.data.cpu().numpy()
        Output_Data.extend(batch_output)
    OutputData = np.array(Output_Data)
    # make shift for -5 s
    # OutPuts = np.concatenate((OutputData[:,5:],np.zeros((len(OutputData),5))),axis=-1)
    # In signal detection tasks, non-shifted version is wanted as the result

h5file.close()
PreFile.close()
end_time=time.time()
print("Prediction_Generated")

toc = end_time-start_time #~1200s 20min
print("Time of Computing",toc)
