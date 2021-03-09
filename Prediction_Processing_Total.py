#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# memory tracking

import argparse
psr = argparse.ArgumentParser()
psr.add_argument('ipt', help='input file')
psr.add_argument('bsl', help='baseline file')
psr.add_argument('gain', help='gaintable file')
psr.add_argument('-o', dest='opt', help='output')
psr.add_argument('-N', dest='NetDir', help='Network directory')
psr.add_argument('--met', type=str, help='method')
psr.add_argument('--ref', type=str, nargs='+', help='reference file')
psr.add_argument('-B', '--batchsize', dest='BAT', type=int, default=15000)
psr.add_argument('-D', '--device', dest='Device', type=str, default='cpu')
args = psr.parse_args()
NetDir = args.NetDir
rawfilename = args.ipt
bslfilename = args.bsl
output = args.opt
BATCHSIZE = args.BAT
reference = args.ref
method = args.met
gaintablefile = args.gain
Device = args.Device

import os
import time
lock_file = "/tmp/Waveformanalysis_lock"
if os.path.exists(lock_file):
    with open(lock_file, "r") as lock :
        process_number = int(lock.readline())
    process_number += 1
    print("process {}".format(process_number))
    with open(lock_file, "w") as lock :
        lock.write("{}".format(process_number))
    if process_number == 1 : time.sleep(10)
    elif process_number == 2 : time.sleep(20)
else :
    with open(lock_file, "w") as lock :
        lock.write("{}".format(0))


global_start = time.time()
cpu_global_start = time.process_time()
import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py
import uproot
import awkward as ak

import torch
torch.no_grad()

if torch.cuda.is_available() and Device == 'cpu':
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    Device = 'cpu'
elif torch.cuda.is_available() : Device = int(Device)
elif Device != 'cpu' :
    print("WARNING: You do not have a CUDA device, so you should probably run without --cuda")
    Device = 'cpu'
else :
    Device = 'cpu'
Device = torch.device(Device)


print('Initialization finished, real time {0:.4f}s, cpu time {1:.4f}s'.format(time.time() - global_start, time.process_time() - cpu_global_start))

# Loading CNN Net
tic = time.time()
nets = dict([])
for channelid in tqdm(range(30), desc='Loading Nets of each channel') :
    nets[channelid] = torch.load(NetDir + '/{:02d}.net'.format(channelid), map_location=Device)
print('Net Loaded, consuming {0:.4f}s'.format(time.time() - tic))


def ReadBaseline(filename) :
    f = uproot.open(filename)
    Ped = ak.flatten(f["SimpleAnalysis"]["ChannelInfo.Pedestal"].array())
    f.close()
    return Ped


def Read_Data(raw_filename, bsl_filename, fileno) :
    RawDataFile = uproot.open(raw_filename)
    ChannelId = RawDataFile["Readout"]["ChannelId"].array()
    TriggerNo = RawDataFile["Readout"]["TriggerNo"].array()
    nchannels = ak.num(ChannelId)
    TriggerNo = np.array(TriggerNo).repeat(nchannels)
    ChannelId = np.array(ak.flatten(ChannelId))
    nwaves = len(ChannelId)
    Waveform = np.array(ak.flatten(RawDataFile["Readout"]["Waveform"].array()))
    WindowSize = int(len(Waveform) / nwaves)
    Waveform = np.array(Waveform).reshape((nwaves, WindowSize))
    waveform_dtype = np.dtype([("TriggerNo", np.uint32), ("ChannelID", np.uint32), ("FileNo", np.int32), ("Pedestal", np.float32), ("Waveform", np.int16, WindowSize)])
    Valid_Channels = ChannelId < 30
    Waves_and_info = np.empty((Valid_Channels).sum(), dtype=waveform_dtype)
    Waves_and_info["TriggerNo"] = TriggerNo[Valid_Channels]
    Waves_and_info["ChannelID"] = ChannelId[Valid_Channels]
    Waves_and_info["Waveform"] = Waveform[Valid_Channels]
    Waves_and_info["Pedestal"] = ReadBaseline(bsl_filename)
    Waves_and_info["FileNo"] = fileno

    wave_dict = {"Waveform" : list(Waves_and_info["Waveform"])}
    for name in waveform_dtype.names :
        if name != "Waveform" : wave_dict.update({name : Waves_and_info[name]})
    RawDataFile.close()
    return pd.DataFrame(wave_dict)


def WriteData(Result, optpath) :
    Result = Result.to_records(index=False)
    with h5py.File(optpath, 'w') as opt:
        dset = opt.create_dataset('AnswerWF', data=Result, compression=32015)
        dset.attrs['Method'] = method


GainTable = np.loadtxt(gaintablefile, skiprows=0, usecols=2)
tic = time.time()
cpu_tic = time.process_time()
fileno = int(rawfilename.split("/")[-1][:-5])
N = 1
Waveforms_and_info = Read_Data(rawfilename, bslfilename, fileno)
print('Data Loaded, consuming {0:.4f}s using {1} threads, cpu time {2:.4f}s'.format(time.time() - tic, N, time.process_time() - cpu_tic))

WindowSize = len(Waveforms_and_info["Waveform"][0:1][0])
Channel_Grouped_Waveform = Waveforms_and_info.groupby(by='ChannelID')

filter_limit = 0.05
Timeline = np.arange(WindowSize, dtype=np.int16).reshape(1, WindowSize)


def Forward(channelid) :
    SPECharge = GainTable[channelid]
    filter_limit = 0.01 * SPECharge
    try :
        Data_of_this_channel = Channel_Grouped_Waveform.get_group(channelid)
    except KeyError :
        print("Warning: channel {0} not found in file {1}!".format(channelid, fileno))
        return None
    Wave = np.vstack(Data_of_this_channel['Waveform'])
    Peds = np.array(Data_of_this_channel['Pedestal'])[:, None]
    TriggerNos = np.array(Data_of_this_channel['TriggerNo'])
    FileNos = np.array(Data_of_this_channel['FileNo'])
    HitPosInWindows = np.empty(0, dtype=np.int16)
    PEmeasure = np.empty(0, dtype=np.float64)
    EventData = np.empty(0, dtype=np.uint32)
    FileNoData = np.empty(0, dtype=np.int32)
    slices = np.append(np.arange(0, len(Wave), BATCHSIZE), len(Wave))
    for i in range(len(slices) - 1) :
        inputs = Peds[slices[i]:slices[i + 1]] - Wave[slices[i]:slices[i + 1]]
        Total = np.abs(np.sum(inputs, axis=1))
        Total = np.where(Total > 1e-4, Total, 1e-4)
        if Device != 'cpu' :
            Prediction = nets[channelid].forward(torch.from_numpy(inputs).to(Device)).data.cpu().numpy()
        else :
            Prediction = nets[channelid].forward(torch.from_numpy(inputs)).data.numpy()
        sumPrediction = np.sum(Prediction, axis=1)
        sumPrediction = np.where(sumPrediction > 1e-4, sumPrediction, 1e-4)
        HitPosInWindow = Prediction > filter_limit
        Prediction = Prediction / sumPrediction[:, None]
        pe_numbers = HitPosInWindow.sum(axis=1)
        no_pe_found = pe_numbers == 0
        if no_pe_found.any() :
            guessed_risetime = inputs[no_pe_found].argmax(axis=1) - 2  # arbitary parameter :2
            HitPosInWindow[no_pe_found, guessed_risetime] = True
            Prediction[no_pe_found, guessed_risetime] = 1
            pe_numbers[no_pe_found] = 1
        Prediction = Prediction / np.sum(Prediction, axis=1)[:, None] * Total[:, None]
        Prediction = Prediction[HitPosInWindow]
        PEmeasure = np.append(PEmeasure, Prediction)
        TimeMatrix = np.repeat(Timeline, len(HitPosInWindow), axis=0)[HitPosInWindow]
        HitPosInWindows = np.append(HitPosInWindows, TimeMatrix)
        EventData = np.append(EventData, np.repeat(TriggerNos[slices[i]:slices[i + 1]], pe_numbers))
        FileNoData = np.append(FileNoData, np.repeat(FileNos[slices[i]:slices[i + 1]], pe_numbers))
        ChannelData = np.empty(EventData.shape, dtype=np.uint32)
        ChannelData.fill(channelid)
    return pd.DataFrame({'HitPosInWindow': HitPosInWindows, 'Charge': PEmeasure, 'TriggerNo': EventData, 'ChannelID': ChannelData, 'FileNo': FileNoData})


tic = time.time()
cpu_tic = time.process_time()
Result = []
for ch in tqdm(range(30), desc='Predict for each channel') :
    Result.append(Forward(ch))
Result = pd.concat(Result)
Result = Result.sort_values(by=['TriggerNo', 'ChannelID'])
print('Prediction generated, real time {0:.4f}s, cpu time {1:.4f}s'.format(time.time() - tic, time.process_time() - cpu_tic))

tic = time.time()
cpu_tic = time.process_time()
WriteData(Result, output)
print('output written, real time {0:.4f}s, cpu time {1:.4f}s'.format(time.time() - tic, time.process_time() - cpu_tic))


print('Finished! Consuming {0:.2f}s in total, cpu time {1:.2f}s.'.format(time.time() - global_start, time.process_time() - cpu_global_start))
