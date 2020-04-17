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

from IPython import embed  # ipython breakpoint inserting
import numpy as np
import numba

import torch
from torch.nn import functional as F

from JPwaptool_Lite import JPwaptool_Lite
from multiprocessing import Pool, Process, Pipe, Lock

import os
import sys
import time

import tables


def seperate_channels(Waveforms_and_info, WindowSize) :
    channel_statics = np.bincount(Waveforms_and_info["ChannelID"])
    Waveform = dict([])
    index = dict([])
    for ch, nWaves in enumerate(channel_statics) :
        if nWaves > 0 :
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
WaveformTable = RawDataFile.root.Waveform
WindowSize = len(WaveformTable[0]['Waveform'])
Total_entries = len(WaveformTable)
print(Total_entries)


def Read_Data(startentry, endentry) :
    return seperate_channels(WaveformTable[startentry:endentry], WindowSize)


tic = time.time()
N = 6
if N == 1 :
    Waveform_set = seperate_channels(WaveformTable[:], WindowSize)
else :
    slices = np.append(np.arange(0, Total_entries, int(np.ceil(Total_entries / N))), Total_entries)
    ranges = list(zip(slices[0:-1], slices[1:]))
    with Pool(N) as pool :
        Waveform_sets = pool.starmap(Read_Data, ranges)
    Waveform_set = Waveform_sets[0]
    for i, WSet in enumerate(Waveform_sets[1:], 1) :
        for ch in WSet :
            Waveform_set[ch]["Wave"] = np.vstack((Waveform_set[ch]["Wave"], WSet[ch]["Wave"]))
            Waveform_set[ch]["Index"] = np.hstack((Waveform_set[ch]["Index"], WSet[ch]["Index"] + slices[i]))
print(N, end=': ')
print(time.time() - tic)


embed()


def Prepare(lock, channelid, downstream) :
    lock.acquire()
    print("lock1 acquired")
    Waves = Waveform_sets[channelid]['Wave']
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
    downstream.send(Shifted_Wave)
    downstream.close()
    lock.release()
    print("lock1 released")


def Forward(upstream, channelid) :
    Shifted_Wave = upstream.recv()
    Hit_Vectors = np.empty(Shifted_Wave.shape, dtype=Shifted_Wave.dtype)
    device = torch.device(1)
    start = time.time()
    net = torch.load(NetDir + "/Channel{}.torch_net".format(channelid), map_location=device)  # Pre-trained Model Parameters
    slices = np.append(np.arange(0, len(Shifted_Wave), BATCHSIZE,), len(Shifted_Wave))
    print(slices)
    for i in range(len(slices) - 1) :
        tensor = torch.from_numpy(Shifted_Wave[slices[i]:slices[i + 1]]).to(device=device)
        Hit_Vectors[slices[i]:slices[i + 1]] = net.forward(tensor).data.cpu().numpy()
    print("consuming {:.4f}s".format(time.time() - start))
    # del tensor
    return Hit_Vectors


process_list = []
result_conn_dict = dict([])
lock = Lock()
for ch in Waveform_sets.keys() :
    parent_conn1, child_conn1 = Pipe()
    p1 = Process(target=Prepare, args=(lock, ch, child_conn1))
    p1.start()
    process_list.append(p1)
    result_conn_dict[ch] = parent_conn1

for ch in result_conn_dict :
    print(Forward(result_conn_dict[ch], ch))
for process in process_list :
    process.join()
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
#filter_limit = 0.9 / WindowSize
#Timeline = torch.arange(WindowSize, device=device).repeat([LoadingPeriod, 1])
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
# h5file.close()
RawDataFile.close()
#end_time = time.time()
# print("Prediction_Generated")
#
# toc = end_time - start_time  # ~1200s 20min
#print("Time of Computing", toc)
