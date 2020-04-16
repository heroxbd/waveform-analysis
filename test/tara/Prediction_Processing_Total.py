import argparse
psr = argparse.ArgumentParser()
psr.add_argument('ipt', help='input file')
psr.add_argument('opt', help='output file')
psr.add_argument('NetDir', help='Network directory')
psr.add_argument('-B', '--batchsize', dest='BAT', type=int, default=20000)
args = psr.parse_args()
NetDir = args.NetDir
outputs = args.opt
filename = args.ipt
BATCHSIZE = args.BAT

# from IPython import embed  # ipython breakpoint inserting
import numpy as np
import numba

import torch
from torch.nn import functional as F

from JPwaptool_Lite import JPwaptool_Lite
from multiprocessing import Pool

import os
import sys
import time

import tables
#device = torch.device('cpu')


def seperate_channels(Waveforms_and_info, WindowSize) :
    channel_statics = np.bincount(Waveforms_and_info["ChannelID"])
    Waveform = dict([])
    index = dict([])
    for ch, nWaves in enumerate(channel_statics) :
        Waveform[ch] = {"Wave": np.empty((nWaves, WindowSize), dtype=np.int16), "Index": np.empty(nWaves, dtype=np.int64)}
        index[ch] = 0
    for i, wave in enumerate(Waveforms_and_info) :
        ch = wave["ChannelID"]
        Waveform[ch]["Wave"][index[ch]] = wave["Waveform"]
        Waveform[ch]["Index"][index[ch]] = i
        index[ch] = index[ch] + 1
    return Waveform


# Loading Data
RawDataFile = tables.open_file(filename, "r")
# Waveforms_and_info = RawDataFile.root.Waveform[:]
Waveforms_and_info = RawDataFile.root.Waveform[0:100000]
WindowSize = len(Waveforms_and_info[0]['Waveform'])
Total_entries = len(Waveforms_and_info)
print(Total_entries)

Waveform_sets = seperate_channels(Waveforms_and_info, WindowSize)


def Forward(channelid) :
    #device = torch.device(channelid%2)
    device = torch.device("cpu")
    Waves = Waveform_sets[channelid]['Wave']
    net = torch.load(NetDir + "/Channel{}.torch_net".format(channelid), map_location=device)  # Pre-trained Model Parameters
    print("net{} loaded".format(channelid))
    if WindowSize >= 1000 :
        stream = JPwaptool_Lite(WindowSize, 100, 600)
    elif WindowSize == 600 :
        stream = JPwaptool_Lite(WindowSize, 50, 400)
    else:
        raise ValueError("Unknown WindowSize, I don't know how to choose the parameters for pedestal calculatation")
    Shifted_Wave = np.empty(Waves.shape, dtype=np.float32)
    for i, w in enumerate(Waves) :
        stream.Calculate(w)
        Shifted_Wave[i] = stream.ChannelInfo.Ped - w
    tensor = torch.from_numpy(Shifted_Wave).to(device=device)
    Shifted_Wave = net.forward(tensor).data.cpu().numpy()
    del tensor
    return Shifted_Wave


#Result = Forward(1)

with Pool(2) as pool :
    Results = pool.map(Forward,range(30))

#
#
## Data Settings
#LoadingPeriod = 20000
## h5 file handling
## Define the database columns
#
#
#class AnswerData(tables.IsDescription):
#    EventID = tables.Int64Col(pos=0)
#    ChannelID = tables.Int16Col(pos=1)
#    PETime = tables.Int16Col(pos=2)
#    Weight = tables.Float32Col(pos=3)
#
#
## Create the output file and the group
## h5file = tables.open_file("./Prediction_Results/Prediction_Mod_ztraining.h5", mode="w", title="OneTonDetector")
#h5file = tables.open_file(outputs, mode="w", title="OneTonDetector", filters=tables.Filters(complevel=9))
#
## Create tables
#AnswerTable = h5file.create_table("/", "Answer", AnswerData, "Answer")
#answer = AnswerTable.row
#
#
## Prepare for data and generating answer from prediction
#filter_limit = 0.9 / WindowSize
#Timeline = torch.arange(WindowSize, device=device).repeat([LoadingPeriod, 1])
#entryList = np.arange(0, Total_entries, LoadingPeriod)
#entryList = np.append(entryList, Total_entries)
#start_time = time.time()
#
#
## Loop for batched data
#for k in range(len(entryList) - 1) :
#    # Making Dataset
#    EventData = Data_set[entryList[k]:entryList[k + 1]]['EventID']
#    ChanData = Data_set[entryList[k]:entryList[k + 1]]['ChannelID']
#    WaveData = Data_set[entryList[k]:entryList[k + 1]]['Waveform']
#    inputs = torch.empty((len(WaveData), WindowSize), device=device)
#    for i in range(len(WaveData)) :
#        stream.Calculate(WaveData[i])
#        inputs[i] = torch.from_numpy(stream.ChannelInfo.Ped - WaveData[i])
#    # Make mark
#    print("Processing entry {0}, Progress {1}%".format(k * LoadingPeriod, k * LoadingPeriod / Total_entries * 100))
#
#    if len(EventData) != len(Timeline) :
#        Timeline = torch.arange(WindowSize, device=device).repeat([len(EventData), 1])
#
#    if k == 0 :
#        if device != 'cpu' :
#            # finish loading to GPU, give tag on .bulletin.swp
#            os.system("echo {} {} >> .bulletin.swp".format(fileno, 0))
#
#    # calculating
#    Prediction = net(inputs).data
#    # checking for no pe event
#    PETimes = Prediction > filter_limit
#    pe_numbers = PETimes.sum(1)
#    no_pe_found = pe_numbers == 0
#    if no_pe_found.any() :
#        print("I cannot find any pe in Event {0}, Channel {1} (entry {2})".format(EventData[no_pe_found.cpu().numpy()], ChanData[no_pe_found.cpu().numpy()], k * LoadingPeriod + np.arange(LoadingPeriod)[no_pe_found.cpu().numpy()]))
#        guessed_petime = F.relu(inputs[no_pe_found].max(1)[1] - 7)
#        PETimes[no_pe_found, guessed_petime] = True
#        Prediction[no_pe_found, guessed_petime] = 1
#        pe_numbers[no_pe_found] = 1
#
#    # Makeing Output and write submission file
#    Weights = Prediction[PETimes].cpu().numpy()
#    PETimes = Timeline[PETimes].cpu().numpy()
#    pe_numbers = pe_numbers.cpu().numpy()
#    EventData = np.repeat(EventData, pe_numbers)
#    ChanData = np.repeat(ChanData, pe_numbers)
#    for i in range(len(PETimes)) :
#        answer['PETime'] = PETimes[i]
#        answer['Weight'] = Weights[i]
#        answer['EventID'] = EventData[i]
#        answer['ChannelID'] = ChanData[i]
#        answer.append()
#
#    # Flush into the output file
#    AnswerTable.flush()
#
#
#h5file.close()
RawDataFile.close()
#end_time = time.time()
#print("Prediction_Generated")
#
#toc = end_time - start_time  # ~1200s 20min
#print("Time of Computing", toc)
