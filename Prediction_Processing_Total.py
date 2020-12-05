# -*- coding: utf-8 -*-

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
psr.add_argument('--cuda'  , action='store_true', help='enables cuda')
args = psr.parse_args()
NetDir = args.NetDir
rawfilename = args.ipt
bslfilename = args.bsl
output = args.opt
BATCHSIZE = args.BAT
reference = args.ref
method = args.met
gaintablefile = args.gain

import time
global_start = time.time()
cpu_global_start = time.process_time()
import numpy as np
import tables
import pandas as pd
from tqdm import tqdm
import h5py
import uproot4
import awkward1 as ak

import torch
from torch import nn
torch.no_grad()

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    Device = 'cpu'
elif torch.cuda.is_available() and args.cuda:
    Device = 'cuda'
elif not torch.cuda.is_available() and args.cuda:
    print("WARNING: You do not have a CUDA device, so you should probably run without --cuda")
    Device = 'cpu'
else :
    Device = 'cpu'

from multiprocessing import Pool, cpu_count
from IPython import embed


def ReadBaseline(filename) :
    f = uproot4.open(filename)
    Ped = ak.flatten(f["SimpleAnalysis"]["ChannelInfo.Pedestal"].array())
    return Ped


def Read_Data(rawfilename, bslfilename) :
    RawDataFile = tables.open_file(rawfilename, 'r')
    WaveformTable = RawDataFile.root.Readout.Waveform
    Waveforms_and_info = WaveformTable[:]
    Shifted_Waves_and_info = np.empty(Waveforms_and_info.shape, dtype=gpufloat_dtype)
    for name in origin_dtype.names :
        if name != 'Waveform' :
            Shifted_Waves_and_info[name] = Waveforms_and_info[name]

    Pedestal = ReadBaseline(bslfilename)
    for i in range(len(Waveforms_and_info)) :
        Shifted_Waves_and_info[i]['Waveform'] = Pedestal[i] - Waveforms_and_info[i]['Waveform'].astype(np.float)
    RawDataFile.close()
    return pd.DataFrame({name: list(Shifted_Waves_and_info[name]) for name in gpufloat_dtype.names})


# Loading Data
RawDataFile = tables.open_file(rawfilename, 'r')
origin_dtype = RawDataFile.root.Readout.Waveform.dtype
Total_entries = len(RawDataFile.root.Readout.Waveform)
RawDataFile.close()
WindowSize = origin_dtype['Waveform'].shape[0]
gpufloat_dtype = np.dtype([(name, np.dtype('float32') if name == 'Waveform' else origin_dtype[name].base, origin_dtype[name].shape) for name in origin_dtype.names])
print('Initialization finished, real time {0:.4f}s, cpu time {1:.4f}s'.format(time.time() - global_start, time.process_time() - cpu_global_start))
print('Processing {} entries'.format(Total_entries))

N = 10
tic = time.time()
cpu_tic = time.process_time()
slices = np.append(np.arange(0, Total_entries, int(np.ceil(Total_entries / N))), Total_entries)
ranges = list(zip(slices[0:-1], slices[1:]))
Waveforms_and_info = Read_Data(rawfilename, bslfilename)
print('Data Loaded, consuming {0:.4f}s using {1} threads, cpu time {2:.4f}s'.format(time.time() - tic, N, time.process_time() - cpu_tic))

channelid_set = set(Waveforms_and_info['ChannelID'])
Channel_Grouped_Waveform = Waveforms_and_info.groupby(by='ChannelID')

# Loading CNN Net
tic = time.time()
nets = dict([])
for channelid in tqdm(channelid_set, desc='Loading Nets of each channel') :
    nets[channelid] = torch.load(NetDir + '/{:02d}.net'.format(channelid))
    if Device == 'cuda' :
        nets[channelid] = nn.DataParallel(nets[channelid], range(torch.cuda.device_count()))

print('Net Loaded, consuming {0:.4f}s'.format(time.time() - tic))

filter_limit = 0.05
Timeline = np.arange(WindowSize).reshape(1, WindowSize)
GainTable = np.loadtxt(gaintablefile, skiprows=0, usecols=2)


def Forward(channelid) :
    SPECharge = GainTable[channelid]
    filter_limit = 0.05 * SPECharge
    Data_of_this_channel = Channel_Grouped_Waveform.get_group(channelid)
    Shifted_Wave = np.vstack(Data_of_this_channel['Waveform'])
    TriggerNos = np.array(Data_of_this_channel['TriggerNo'])
    HitPosInWindows = np.empty(0, dtype=np.int16)
    PEmeasure = np.empty(0, dtype=np.float32)
    EventData = np.empty(0, dtype=np.int64)
    slices = np.append(np.arange(0, len(Shifted_Wave), BATCHSIZE), len(Shifted_Wave))
    for i in range(len(slices) - 1) :
        inputs = Shifted_Wave[slices[i]:slices[i + 1]]
        Total = np.abs(np.sum(inputs, axis=1))
        Total = np.where(Total > 1e-4, Total, 1e-4)
        if Device == 'cuda' :
            Prediction = nets[channelid].forward(torch.from_numpy(inputs).cuda()).data.cpu().numpy()
        else :
            Prediction = nets[channelid].forward(torch.from_numpy(inputs)).data.numpy()
        sumPrediction = np.sum(Prediction, axis=1)
        sumPrediction = np.where(sumPrediction > 1e-4, sumPrediction, 1e-4)
        Prediction = Prediction / sumPrediction[:, None]
        HitPosInWindow = Prediction > filter_limit
        pe_numbers = HitPosInWindow.sum(axis=1)
        no_pe_found = pe_numbers == 0
        if no_pe_found.any() :
            guessed_risetime = inputs[no_pe_found].argmax(axis=1) - 2 #  arbitary parameter :2
            HitPosInWindow[no_pe_found, guessed_risetime] = True
            pe_numbers[no_pe_found] = 1
        Prediction = np.where(Prediction > filter_limit, Prediction, 0)
        Prediction = Prediction / np.sum(Prediction, axis=1)[:, None] * Total[:, None]
        Prediction = Prediction * SPECharge
        Prediction = Prediction[HitPosInWindow]
        PEmeasure = np.append(PEmeasure, Prediction)
        TimeMatrix = np.repeat(Timeline, len(HitPosInWindow), axis=0)[HitPosInWindow]
        HitPosInWindows = np.append(HitPosInWindows, TimeMatrix)
        EventData = np.append(EventData, np.repeat(TriggerNos[slices[i]:slices[i + 1]], pe_numbers))
        ChannelData = np.empty(EventData.shape, dtype=np.int16)
        ChannelData.fill(channelid)
    return pd.DataFrame({'HitPosInWindow': HitPosInWindows, 'Charge': PEmeasure, 'TriggerNo': EventData, 'ChannelID': ChannelData})


tic = time.time()
cpu_tic = time.process_time()
Result = []
for ch in tqdm(channelid_set, desc='Predict for each channel') :
    Result.append(Forward(ch))
Result = pd.concat(Result)
Result = Result.sort_values(by=['TriggerNo', 'ChannelID'])
Result = Result.to_records(index=False)
print('Prediction generated, real time {0:.4f}s, cpu time {1:.4f}s'.format(time.time() - tic, time.process_time() - cpu_tic))

with h5py.File(output, 'w') as opt:
    dset = opt.create_dataset('Answer', data=Result, compression='gzip')
    dset.attrs['Method'] = method
    print('The output file path is {}'.format(output))

print('Finished! Consuming {0:.2f}s in total, cpu time {1:.2f}s.'.format(time.time() - global_start, time.process_time() - cpu_global_start))
