# -*- coding: utf-8 -*-

import argparse
psr = argparse.ArgumentParser()
psr.add_argument('ipt', help='input file')
psr.add_argument('-o', dest='opt', help='output')
psr.add_argument('-N', dest='NetDir', help='Network directory')
psr.add_argument('--mod', type=str, help='mode of weight', choices=['PEnum', 'Charge'])
psr.add_argument('--met', type=str, help='method')
psr.add_argument('--ref', type=str, nargs='+', help='reference file')
psr.add_argument('-B', '--batchsize', dest='BAT', type=int, default=15000)
psr.add_argument('-D', '--device', dest='Device', type=str, default='cpu')
args = psr.parse_args()
NetDir = args.NetDir
filename = args.ipt
output = args.opt
BATCHSIZE = args.BAT
Device = args.Device
reference = args.ref
method = args.met
mode = args.mod

import time
global_start = time.time()
cpu_global_start = time.process_time()
import numpy as np
import tables
import pandas as pd
from tqdm import tqdm
from JPwaptool import JPwaptool
import h5py

import torch
from torch.nn import functional as F

from multiprocessing import Pool, cpu_count
import wf_func as wff

def Read_Data(startentry, endentry) :
    RawDataFile = tables.open_file(filename, 'r')
    WaveformTable = RawDataFile.root.Waveform
    stream = JPwaptool(WindowSize, 150, 600, 7, 15)
    Waveforms_and_info = WaveformTable[startentry:endentry]
    Shifted_Waves_and_info = np.empty(Waveforms_and_info.shape, dtype=gpufloat_dtype)
    for name in origin_dtype.names :
        if name != 'Waveform' :
            Shifted_Waves_and_info[name] = Waveforms_and_info[name]
    for i in range(len(Waveforms_and_info)) :
        channelid = Waveforms_and_info[i]['ChannelID']
        stream.Calculate(Waveforms_and_info[i]['Waveform'])
        Shifted_Waves_and_info[i]['Waveform'] = (Waveforms_and_info[i]['Waveform'] - stream.ChannelInfo.Pedestal) * spe_pre[channelid]['epulse']
    RawDataFile.close()
    return pd.DataFrame({name: list(Shifted_Waves_and_info[name]) for name in gpufloat_dtype.names})

# Loading Data
RawDataFile = tables.open_file(filename, 'r')
origin_dtype = RawDataFile.root.Waveform.dtype
Total_entries = len(RawDataFile.root.Waveform)
RawDataFile.close()
WindowSize = origin_dtype['Waveform'].shape[0]
gpufloat_dtype = np.dtype([(name, np.dtype('float32') if name == 'Waveform' else origin_dtype[name].base, origin_dtype[name].shape) for name in origin_dtype.names])
print('Initialization finished, real time {0:.4f}s, cpu time {1:.4f}s'.format(time.time() - global_start, time.process_time() - cpu_global_start))
print('Processing {} entries'.format(Total_entries))

N = 10
tic = time.time()
cpu_tic = time.process_time()
spe_pre = wff.read_model(reference[0])
slices = np.append(np.arange(0, Total_entries, int(np.ceil(Total_entries / N))), Total_entries)
ranges = list(zip(slices[0:-1], slices[1:]))
with Pool(min(N, cpu_count())) as pool :
    Waveforms_and_info = pd.concat(pool.starmap(Read_Data, ranges))
print('Data Loaded, consuming {0:.4f}s using {1} threads, cpu time {2:.4f}s'.format(time.time() - tic, N, time.process_time() - cpu_tic))

channelid_set = set(Waveforms_and_info['ChannelID'])
Channel_Grouped_Waveform = Waveforms_and_info.groupby(by='ChannelID')

# Loading CNN Net
tic = time.time()
Device = int(Device)
device = torch.device(Device)
nets = dict([])
for channelid in tqdm(channelid_set, desc='Loading Nets of each channel') :
    nets[channelid] = torch.load(NetDir + '/Channel{:02d}.torch_net'.format(channelid), map_location=device)
print('Net Loaded, consuming {0:.4f}s'.format(time.time() - tic))

filter_limit = 0.9 / WindowSize
Timeline = np.arange(WindowSize).reshape(1, WindowSize)

def Forward(channelid) :
    SPECharge = spe_pre[channelid]['spe'].sum()
    Data_of_this_channel = Channel_Grouped_Waveform.get_group(channelid)
    Shifted_Wave = np.vstack(Data_of_this_channel['Waveform'])
    EventIDs = np.array(Data_of_this_channel['EventID'])
    RiseTimes = np.empty(0, dtype=np.int16)
    PEmeasure = np.empty(0, dtype=np.float32)
    EventData = np.empty(0, dtype=np.int64)
    slices = np.append(np.arange(0, len(Shifted_Wave), BATCHSIZE), len(Shifted_Wave))
    for i in range(len(slices) - 1) :
        inputs = Shifted_Wave[slices[i]:slices[i + 1]]
        Total = np.abs(np.sum(inputs, axis=1)) / SPECharge
        Total = np.where(Total > 0.1, Total, 0.1)
        if mode == 'Charge':
            Total = Total * SPECharge
        Prediction = np.abs(nets[channelid].forward(torch.from_numpy(inputs).to(device=device)).data.cpu().numpy())
        RiseTime = Prediction > filter_limit
        pe_numbers = RiseTime.sum(axis=1)
        no_pe_found = pe_numbers == 0
        if no_pe_found.any() :
            # print('Cannot find any pe in Event {0}, Channel {1}'.format(EventIDs[slices[i]:slices[i + 1]][no_pe_found.cpu().numpy()], channelid))
            guessed_risetime = np.around(inputs[no_pe_found].argmax(axis=1) - spe_pre[channelid]['peak_c'])
            guessed_risetime = np.where(guessed_risetime > 0, guessed_risetime, 0)
            RiseTime[no_pe_found, guessed_risetime] = True
            Prediction[no_pe_found, guessed_risetime] = 1
            pe_numbers[no_pe_found] = 1
        Prediction = np.abs((Prediction / np.sum(Prediction, axis=1)[:,None] * Total[:,None])[RiseTime])
        PEmeasure = np.append(PEmeasure, Prediction)
        TimeMatrix = np.repeat(Timeline, len(RiseTime), axis=0)[RiseTime]
        RiseTimes = np.append(RiseTimes, TimeMatrix)
        EventData = np.append(EventData, np.repeat(EventIDs[slices[i]:slices[i + 1]], pe_numbers))
        ChannelData = np.empty(EventData.shape, dtype=np.int16)
        ChannelData.fill(channelid)
    return pd.DataFrame({'RiseTime': RiseTimes, mode: PEmeasure, 'EventID': EventData, 'ChannelID': ChannelData})

tic = time.time()
cpu_tic = time.process_time()
Result = []
for ch in tqdm(channelid_set, desc='Predict for each channel') :
    Result.append(Forward(ch))
Result = pd.concat(Result)
Result = Result.sort_values(by=['EventID', 'ChannelID'])
Result = Result.to_records(index=False)
print('Prediction generated, real time {0:.4f}s, cpu time {1:.4f}s'.format(time.time() - tic, time.process_time() - cpu_tic))

with h5py.File(output, 'w') as opt:
    dset = opt.create_dataset('Answer', data=Result, compression='gzip')
    dset.attrs['Method'] = method
    print('The output file path is {}'.format(output))

print('Finished! Consuming {0:.2f}s in total, cpu time {1:.2f}s.'.format(time.time() - global_start, time.process_time() - cpu_global_start))
