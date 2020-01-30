from IPython import embed #ipython breakpoint inserting
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
    
# use cpu or gpu
device = torch.device(2)

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
net = torch.load(NetDir+net_name)#Pre-trained Model Parameters
if device==torch.device('cpu') : net=net.cpu()
else : net=net.cuda(device)

# Data Settings
LoadingPeriod= 4000
# h5 file handling
# Define the database columns
class AnswerData(tables.IsDescription):
    EventID    = tables.Int64Col(pos=0)
    ChannelID  = tables.Int16Col(pos=1)
    PETime     = tables.Int16Col(pos=2)
    Weight     = tables.Float32Col(pos=3)

# Create the output file and the group
# h5file = tables.open_file("./Prediction_Results/Prediction_Mod_ztraining.h5", mode="w", title="OneTonDetector")
h5file = tables.open_file(SavePath+"Prediction.h5", mode="w", title="OneTonDetector",filters=tables.Filters(complevel=9))

# Create tables
AnswerTable = h5file.create_table("/", "Answer", AnswerData, "Answer")
answer = AnswerTable.row

# Loading Data
PreFile =  tables.open_file(LoadPath+"Pre.h5")
Data_set = PreFile.root.TestDataTable
WindowSize = len(Data_set[0]['Waveform'])
Total_entries = len(Data_set)
print(Total_entries)

# Prepare for data and generating answer from prediction
filter_limit = 0.9/WindowSize 
Timeline = torch.arange(WindowSize,device=device).repeat([LoadingPeriod,1])
entryList = np.arange(0,Total_entries,LoadingPeriod)
entryList = np.append(entryList,Total_entries)
start_time = time.time()
# Loop for batched data
for k in range(len(entryList)-1) :
    # Making Dataset
    EventData = Data_set[entryList[k]:entryList[k+1]]['EventID']
    ChanData = Data_set[entryList[k]:entryList[k+1]]['ChannelID']
    WaveData = Data_set[entryList[k]:entryList[k+1]]['Waveform']
    inputs = torch.tensor(WaveData,device=device).float()

    # Make mark
    print("Processing entry {0}, Progress {1}%".format(k*LoadingPeriod,k*LoadingPeriod/Total_entries*100))
    
    if len(EventData)!=len(Timeline) : 
        Timeline = torch.arange(WindowSize,device=device).repeat([len(EventData),1])
                    
    #calculating
    Prediction = net(inputs).data
    # checking for no pe event
    PETimes = Prediction>filter_limit
    pe_numbers = PETimes.sum(1)
    no_pe_found = pe_numbers==0 
    if no_pe_found.any() :
        print("I cannot find any pe in Event {0}, Channel {1} (entry {2})".format(EventData[no_pe_found.cpu().numpy()],ChanData[no_pe_found.cpu().numpy()],k*LoadingPeriod+np.arange(LoadingPeriod)[no_pe_found.cpu().numpy()]))
        guessed_petime = F.relu(inputs[no_pe_found].max(1)[1]-7)
        PETimes[no_pe_found,guessed_petime] = True
        Prediction[no_pe_found,guessed_petime] = 1
        pe_numbers[no_pe_found] = 1
    
    
    # Makeing Output and write submission file
    Weights = Prediction[PETimes].cpu().numpy()
    PETimes = Timeline[PETimes].cpu().numpy()
    pe_numbers = pe_numbers.cpu().numpy()
    EventData = np.repeat(EventData,pe_numbers)
    ChanData = np.repeat(ChanData,pe_numbers)
    for i in range(len(PETimes)) :
        answer['PETime'] = PETimes[i]
        answer['Weight'] = Weights[i]
        answer['EventID'] = EventData[i]
        answer['ChannelID'] = ChanData[i]
        answer.append()
        
    # Flush into the output file
    AnswerTable.flush()


h5file.close()
PreFile.close()
end_time=time.time()
print("Prediction_Generated")

toc = end_time-start_time #~1200s 20min
print("Time of Computing",toc)
