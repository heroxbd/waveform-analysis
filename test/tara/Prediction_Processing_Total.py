#!/usr/bin/env python3
import argparse
psr = argparse.ArgumentParser()
psr.add_argument('ipt', help='input file')
psr.add_argument('opt', help='output file')
psr.add_argument('NetDir', help='Network directory')
psr.add_argument('-B', '--batchsize', dest='BAT', type=int, default=10000)
args = psr.parse_args()
NetDir = args.NetDir
outputs = args.opt
filename = args.ipt
BATCHSIZE = args.BAT

import time
from IPython import embed  # ipython breakpoint inserting
import numpy as np
import tables
import pandas as pd
# import numba

import torch
from torch.nn import functional as F

from multiprocessing import Pool, Process, Pipe, Lock
from JPwaptool_Lite import JPwaptool_Lite

# Loading Data
RawDataFile = tables.open_file(filename, "r")
WaveformTable = RawDataFile.root.Waveform
Total_entries = len(WaveformTable)
print(Total_entries)


def Read_Data(startentry, endentry) :
    return WaveformTable[startentry:endentry]


N = 6
tic = time.time()
if N == 1 :
    Waveforms_and_info = WaveformTable[:]
else :
    slices = np.append(np.arange(0, Total_entries, int(np.ceil(Total_entries / N))), Total_entries)
    ranges = list(zip(slices[0:-1], slices[1:]))
    with Pool(N) as pool :
        Waveforms_and_info = np.hstack(pool.starmap(Read_Data, ranges))
Waveforms_and_info = pd.DataFrame({name: list(Waveforms_and_info[name]) for name in Waveforms_and_info.dtype.names})
print(N, end=': ')
print(time.time() - tic)

WindowSize = len(Waveforms_and_info['Waveform'][0])
channelid_set = set(Waveforms_and_info['ChannelID'])
Channel_Grouped_Waveform = Waveforms_and_info.groupby(by="ChannelID")


def Prepare(lock, channelid, downstream) :
    lock.acquire()
    Data_of_this_channel = Channel_Grouped_Waveform.get_group(channelid)
    Waves = np.vstack(Data_of_this_channel['Waveform'])
    EventIDs = np.array(Data_of_this_channel['Waveform'])
    Shifted_Wave = np.empty(Waves.shape, dtype=np.float32)
    if WindowSize >= 1000 :
        stream = JPwaptool_Lite(WindowSize, 100, 600)
    elif WindowSize == 600 :
        stream = JPwaptool_Lite(WindowSize, 50, 400)
    else:
        raise ValueError("Unknown WindowSize, I don't know how to choose the parameters for pedestal calculatation")
    for i, w in enumerate(Waves) :
        stream.Calculate(w)
        Shifted_Wave[i] = stream.ChannelInfo.Ped - w
    print("channel {} Prepared".format(ch))
    downstream.send({"Shifted_Wave": Shifted_Wave, "EventIDs": EventIDs})
    downstream.close()
    lock.release()


device = torch.device(1)
filter_limit = 0.9 / WindowSize
Timeline = torch.arange(WindowSize, device=device)


def Forward(upstream, channelid) :
    net = torch.load(NetDir + "/Channel{}.torch_net".format(channelid), map_location=device)  # Pre-trained Model Parameters
    upstream = upstream.recv()
    Shifted_Wave = upstream["Shifted_Wave"]
    EventIDs = upstream["EventIDs"]
    PETimes = np.empty(0, dtype=np.int16)
    Weights = np.empty(0, dtype=np.float32)
    EventData = np.empty(0, dtype=np.int64)
    slices = np.append(np.arange(0, len(Shifted_Wave), BATCHSIZE,), len(Shifted_Wave))
    for i in range(len(slices) - 1) :
        inputs = torch.from_numpy(Shifted_Wave[slices[i]:slices[i + 1]])
        Prediction = net.forward(inputs.to(device=device)).data
        PETime = Prediction > filter_limit
        pe_numbers = PETime.sum(1)
        no_pe_found = (pe_numbers == 0)
        if no_pe_found.any() :
            print("I cannot find any pe in Event {0}, Channel {1}".format(EventIDs[slices[i]:slices[i + 1]][no_pe_found.cpu().numpy()], channelid))
            guessed_petime = F.relu(inputs[no_pe_found].max(1)[1] - 7)
            PETime[no_pe_found, guessed_petime] = True
            Prediction[no_pe_found, guessed_petime] = 1
            pe_numbers[no_pe_found] = 1
        Weights = np.append(Weights, Prediction[PETime].cpu().numpy())
        TimeMatrix = Timeline.repeat([len(PETime), 1])[PETime]
        PETimes = np.append(PETimes, TimeMatrix.cpu().numpy())
        pe_numbers = pe_numbers.cpu().numpy()
        EventData = np.append(EventData, np.repeat(EventIDs[slices[i]:slices[i + 1]], pe_numbers))
    return {"PETime": PETimes, "Weight": Weights, "EventID": EventData}


process_list = []
result_conn_dict = dict([])
lock = Lock()
for ch in channelid_set :
    parent_conn1, child_conn1 = Pipe()
    p1 = Process(target=Prepare, args=(lock, ch, child_conn1))
    p1.start()
    process_list.append(p1)
    result_conn_dict[ch] = parent_conn1

Result = dict([])
for ch in result_conn_dict :
    Result[ch] = (Forward(result_conn_dict[ch], ch))
for process in process_list :
    process.join()

embed()

#parent_conn1, child_conn1 = Pipe()
#parent_conn2, child_conn2 = Pipe()
#p1 = Process(target=Prepare, args=(child_conn1, 0))
# p1.start()
#p2 = Process(target=Forward, args=(child_conn2, parent_conn1.recv(), 0))
# p2.start()
#print("p2 start")
# p1.join()
#print("p1 finished")
# p2.join()
#print("p2 finished")
# print(parent_conn2.recv())
# with Pool(2) as pool :
#    Results = pool.map(Forward,range(30))
#
#
# Data Settings
#LoadingPeriod = 20000
# h5 file handling
# Define the database columns
#
#
# class AnswerData(tables.IsDescription):
#    EventID = tables.Int64Col(pos=0)
#    ChannelID = tables.Int16Col(pos=1)
#    PETime = tables.Int16Col(pos=2)
#    Weight = tables.Float32Col(pos=3)
#
#
# Create the output file and the group
## h5file = tables.open_file("./Prediction_Results/Prediction_Mod_ztraining.h5", mode="w", title="OneTonDetector")
#h5file = tables.open_file(outputs, mode="w", title="OneTonDetector", filters=tables.Filters(complevel=9))
#
# Create tables
#AnswerTable = h5file.create_table("/", "Answer", AnswerData, "Answer")
#answer = AnswerTable.row
#
#
# Prepare for data and generating answer from prediction
#entryList = np.arange(0, Total_entries, LoadingPeriod)
#entryList = np.append(entryList, Total_entries)
#start_time = time.time()
#
#
# Loop for batched data
# for k in range(len(entryList) - 1) :
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
#
#    # Makeing Output and write submission file
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
# h5file.close()
RawDataFile.close()
#end_time = time.time()
# print("Prediction_Generated")
#
# toc = end_time - start_time  # ~1200s 20min
#print("Time of Computing", toc)
