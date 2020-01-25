import numpy as np

import torch
import torch.utils.data as Data

from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

import os,sys
import time

import tables


BATCHSIZE=16

# Make Saving_Directory
SavePath = sys.argv[1]
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

net = torch.load(sys.argv[3]) # Pre-trained Model Parameters

# Data Settings
LoadPath= sys.argv[2]
LoadingPeriod= 200000
# h5 file handling
# Define the database columns
class AnswerData(tables.IsDescription):
    EventID    = tables.Int64Col(pos=0)
    ChannelID  = tables.Int16Col(pos=1)
    PETime     = tables.Int16Col(pos=2)
    Weight     = tables.Float32Col(pos=3)

# Create the output file and the group
# h5file = tables.open_file("./Prediction_Results/Prediction_Mod_ztraining.h5", mode="w", title="OneTonDetector")
h5file = tables.open_file("./Prediction_Results/Prediction_Mod_{}.h5".format(sys.argv[2]), mode="w", title="OneTonDetector")


# Create tables
AnswerTable = h5file.create_table("/", "Answer", AnswerData, "Answer")
answer = AnswerTable.row


start_time = time.time()

# Loading Data
PreFile =  tables.open_file(LoadPath+"Pre.h")
Data_set = PreFile.root.TestDataTable
WindowSize = len(Data_set['Wave'][0])
print(fullname)
Total_entries = len(Data_set)
print(Total_entries)

entryList = np.arange(Total_entries,LoadingPeriod)
entryList = np.append(entryList,Total_entries)
for k,entry in enumerate(entryList) :
    EventData = Data_set[entry:entryList(k+1)]['Event']
    ChanData = Data_set[entry:entryList(k+1)]['Chan']
    WaveData = Data_set[entry:entryList(k+1)]['Wave']
    # Making Dataset
    predict_data = torch.from_numpy(WaveData).cuda(device=2).float()
    predict_loader = Data.DataLoader(dataset=predict_data,batch_size=BATCHSIZE,shuffle=False)

    # Makeing Output
    Output_Data = []
    for i,data in enumerate(predict_loader,0):
        inputs = Variable(data)
        outputs = net(inputs)
        batch_output = outputs.data.cpu().numpy()
        Output_Data.extend(batch_output)
    OutputData = np.array(Output_Data)
    # make shift for -5 s
    # OutPuts = np.concatenate((OutputData[:,5:],np.zeros((len(OutputData),5))),axis=-1)
    # In signal detection tasks, non-shifted version is wanted as the result

    # Write data
    filter_limit = 0.9/WindowSize
    for j in range(len(OutputData)):   # OutputData
        Prediction=OutputData[j]       # OutputData
        EventID=EventData[j]
        ChannelID=ChanData[j]
        # if EventID == 13381 and ChannelID == 29:
        #     print('appear') ??
        if np.sum(Prediction) <= 0:
            Prediction = np.ones(WindowSize) / WindowSize
            print("warning")
        numPE = 0
        for k in range(len(Prediction)):
            if Prediction[k]>filter_limit:
                answer['EventID'] = EventID
                answer['ChannelID'] = ChannelID
                answer['PETime'] = k
                answer['Weight'] = Prediction[k]
                answer.append()
                numPE += 1
        if numPE == 0 :
            answer['EventID'] = EventID
            answer['ChannelID'] = ChannelID
            answer['PETime'] = 300
            answer['Weight'] = 1
            answer.append()
            print(EventID,ChannelID)

        # Make mark
        if (j+1) % 10000 == 0:
            print(j+1)
            
    # Flush into the output file
    AnswerTable.flush()


h5file.close()
end_time=time.time()
print("Prediction_Generated")

toc = end_time-start_time #~1200s 20min
print("Time of Computing",toc)



















