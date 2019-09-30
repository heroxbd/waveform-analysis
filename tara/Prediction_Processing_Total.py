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
SavePath = "./Prediction_Results"
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

net = torch.load("./Network_Models/ftraining-0_0-199999_epoch24_loss0.9691") # Pre-trained Model Parameters

# Data Settings
LoadPath= "./{}/Prediction_Pre-Processing_Results/".format(sys.argv[2])
SavingPeriod= 200000
DataNameVec = []
# Make Name Group
'''
for count in range(8):
    start= count*SavingPeriod
    end = (count+1)*SavingPeriod-1
    name = str(start)+'-'+str(end)
    DataNameVec.append(name)

#DataNameVec.append('1600000-1605329')
#DataNameVec.append('1200000-1341564')
DataNameVec.append('1600000-1796735')

FullNameVec=[]
for name in DataNameVec:
    FullName = LoadPath + 'prediction_' + name + '.npz'
    FullNameVec.append(FullName)
'''
FullNameVec = [LoadPath+i for i in os.listdir(LoadPath)]
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
# Start Loop on Files:
for fullname in FullNameVec:
    # Loading Data
    Data_set = np.load(fullname)
    EventData = Data_set['Event']
    ChanData = Data_set['Chan']
    WaveData = Data_set['Wave']
    print(fullname)
    print("Data_loaded")
    print(len(EventData),len(WaveData),len(WaveData))

    # Making Dataset
    predict_data = torch.from_numpy(WaveData).float()
    predict_loader = Data.DataLoader(dataset=predict_data,batch_size=BATCHSIZE,shuffle=False)

    # Makeing Output
    Output_Data = []
    for i,data in enumerate(predict_loader,0):
        inputs = Variable(data)
        outputs = net(inputs)
        batch_output = outputs.data.numpy()
        Output_Data.extend(batch_output)
    OutputData = np.array(Output_Data)
    # make shift for -5 s
    # OutPuts = np.concatenate((OutputData[:,5:],np.zeros((len(OutputData),5))),axis=-1)
    # In signal detection tasks, non-shifted version is wanted as the result

    # Write data
    filter_limit = 0 #5e-5
    for j in range(len(OutputData)):   # OutputData
        Prediction=OutputData[j]       # OutputData
        EventID=EventData[j]
        ChannelID=ChanData[j]
        if EventID == 13381 and ChannelID == 29:
            print('appear')
        if np.sum(Prediction) <= 0:
            Prediction = np.ones(1029) / 10000
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
end_time=time.time()

AnswerTable.flush()
h5file.close()
print("Prediction_Generated")

toc = end_time-start_time #~1200s 20min
print("Time of Computing",toc)



















