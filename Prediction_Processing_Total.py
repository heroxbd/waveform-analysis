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

import time
global_start = time.time()
cpu_global_start = time.process_time()
import numpy as np
import tables
import pandas as pd
from tqdm import tqdm
import h5py
import uproot
import awkward1 as ak

import torch
from torch import nn
torch.no_grad()

from multiprocessing import Pool, cpu_count
from IPython import embed


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


import os
import re
filenames = os.listdir(rawfilename)
pattern = re.compile(r"(\d+)\.root")
filenos = [int(pattern.match(fn)[1]) for fn in filenames]
print('Initialization finished, real time {0:.4f}s, cpu time {1:.4f}s'.format(time.time() - global_start, time.process_time() - cpu_global_start))

# Loading CNN Net
tic = time.time()
nets = dict([])
for channelid in tqdm(range(30), desc='Loading Nets of each channel') :
    nets[channelid] = torch.load(NetDir + '/{:02d}.net'.format(channelid))
    if Device != 'cpu' :
        nets[channelid] = nn.DataParallel(nets[channelid], [Device])


def ReadBaseline(filename) :
    f = uproot.open(filename)
    Ped = ak.flatten(f["SimpleAnalysis"]["ChannelInfo.Pedestal"].array())
    return Ped


def Read_Data(fileno) :
    raw_filename = rawfilename + "/{}.root".format(fileno)
    bsl_filename = bslfilename + "/{}.root".format(fileno)
    RawDataFile = uproot.open(raw_filename)
    ChannelId = RawDataFile["Readout"]["ChannelId"].array()
    TriggerNo = RawDataFile["Readout"]["TriggerNo"].array()
    nchannels = ak.num(ChannelId)
    TriggerNo = np.array(TriggerNo).repeat(nchannels)
    ChannelId = np.array(ak.flatten(ChannelId))
    nwaves = len(ChannelId)
    Waveform = RawDataFile["Readout"]["Waveform"].array(flatten=True)
    WindowSize = int(len(Waveform) / nwaves)
    Waveform = np.array(Waveform).reshape((nwaves, WindowSize))
    waveform_dtype = np.dtype([("TriggerNo", np.int64), ("ChannelID", np.int16), ("FileNo", np.int32), ("Pedestal", np.float32), ("Waveform", np.int16, WindowSize)])
    Waves_and_info = np.empty(nwaves, dtype=waveform_dtype)
    Valid_Channels = ChannelId < 30
    Waves_and_info["TriggerNo"] = TriggerNo[Valid_Channels]
    Waves_and_info["ChannelID"] = ChannelId[Valid_Channels]
    Waves_and_info["Waveform"] = Waveform[Valid_Channels]
    Waves_and_info["Pedestal"] = ReadBaseline(bsl_filename)
    Waves_and_info["FileNo"] = fileno
    print(rawfilename + " " + bslfilename + " {}".format(fileno))

    wave_dict = {"Waveform" : list(Waves_and_info["Waveform"])}
    for name in waveform_dtype.names :
        if name != "Waveform" : wave_dict.update({name : Waves_and_info[name]})
    return pd.DataFrame(wave_dict)


N = 10
SLICES = np.append(np.arange(0, len(filenos), N), len(filenos))
for i in range(len(SLICES) - 1):
    FILENOS = filenos[SLICES[i]:SLICES[i + 1]]
    # Loading Data
    tic = time.time()
    cpu_tic = time.process_time()
    # files = [(rawfilename + "/{}.root".format(i), bslfilename + "/{}.root".format(i), i) for i in range(3)]
    with Pool(min(N, cpu_count())) as pool :
        Waveforms_and_info = pd.concat(pool.map(Read_Data, FILENOS))
    # Waveforms_and_info = pd.concat([Read_Data(rawfilename + "/{}.root".format(i), bslfilename + "/{}.root".format(i), i) for i in range(3)])

    # Waveforms_and_info = Read_Data(rawfilename, bslfilename)
    print('Data Loaded, consuming {0:.4f}s using {1} threads, cpu time {2:.4f}s'.format(time.time() - tic, N, time.process_time() - cpu_tic))
    print('Processing {} waves'.format(len(Waveforms_and_info)))
    WindowSize = len(Waveforms_and_info["Waveform"][0:1][0])

    Channel_Grouped_Waveform = Waveforms_and_info.groupby(by='ChannelID')

    print('Net Loaded, consuming {0:.4f}s'.format(time.time() - tic))

    filter_limit = 0.05
    Timeline = np.arange(WindowSize).reshape(1, WindowSize)
    GainTable = np.loadtxt(gaintablefile, skiprows=0, usecols=2)

    def Forward(channelid) :
        SPECharge = GainTable[channelid]
        filter_limit = 0.01 * SPECharge
        Data_of_this_channel = Channel_Grouped_Waveform.get_group(channelid)
        Wave = np.vstack(Data_of_this_channel['Waveform'])
        Peds = np.array(Data_of_this_channel['Pedestal'])[:, None]
        TriggerNos = np.array(Data_of_this_channel['TriggerNo'])
        FileNos = np.array(Data_of_this_channel['FileNo'])
        HitPosInWindows = np.empty(0, dtype=np.int16)
        PEmeasure = np.empty(0, dtype=np.float32)
        EventData = np.empty(0, dtype=np.int64)
        FileNoData = np.empty(0, dtype=np.int32)
        slices = np.append(np.arange(0, len(Wave), BATCHSIZE), len(Wave))
        for i in range(len(slices) - 1) :
            inputs = Peds[slices[i]:slices[i + 1]] - Wave[slices[i]:slices[i + 1]]
            Total = np.abs(np.sum(inputs, axis=1))
            Total = np.where(Total > 1e-4, Total, 1e-4)
            if Device != 'cuda' :
                Prediction = nets[channelid].forward(torch.from_numpy(inputs).cuda()).data.cpu().numpy()
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
            ChannelData = np.empty(EventData.shape, dtype=np.int16)
            ChannelData.fill(channelid)
        return pd.DataFrame({'HitPosInWindow': HitPosInWindows, 'Charge': PEmeasure, 'TriggerNo': EventData, 'ChannelID': ChannelData, 'FileNo': FileNoData})

    tic = time.time()
    cpu_tic = time.process_time()
    Result = []
    for ch in tqdm(range(30), desc='Predict for each channel') :
        Result.append(Forward(ch))
    Result = pd.concat(Result)
    Result = Result.sort_values(by=['FileNo', 'TriggerNo', 'ChannelID'])
    Grouped_Result = Result.groupby(by='FileNo')

    def WriteData(fileno) :
        Result = Grouped_Result.get_group(fileno)
        Result = Result.to_records(index=False)
        with h5py.File(output + "/{}.h5".format(fileno), 'w') as opt:
            dset = opt.create_dataset('Answer', data=Result, compression='gzip')
            dset.attrs['Method'] = method
            print('The output file path is {}'.format(output))

    with Pool(N) as pool :
        pool.map(WriteData, FILENOS)
    print('Prediction generated, real time {0:.4f}s, cpu time {1:.4f}s'.format(time.time() - tic, time.process_time() - cpu_tic))

print('Finished! Consuming {0:.2f}s in total, cpu time {1:.2f}s.'.format(time.time() - global_start, time.process_time() - cpu_global_start))
