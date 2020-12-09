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
import gc

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
    nets[channelid] = torch.load(NetDir + '/{:02d}.net'.format(channelid), map_location=Device)
print('Net Loaded, consuming {0:.4f}s'.format(time.time() - tic))


def ReadBaseline(filename) :
    f = uproot.open(filename)
    Ped = ak.flatten(f["SimpleAnalysis"]["ChannelInfo.Pedestal"].array())
    f.close()
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
    waveform_dtype = np.dtype([("TriggerNo", np.uint32), ("ChannelID", np.uint32), ("FileNo", np.int32), ("Pedestal", np.float32), ("Waveform", np.int16, WindowSize)])
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
    RawDataFile.close()
    return pd.DataFrame(wave_dict)


def WriteData(fileno, Grouped_Result) :
    optpath = output + "/{}.h5".format(fileno)
    Result = Grouped_Result.get_group(fileno)
    Result = Result.to_records(index=False)
    with h5py.File(optpath, 'w') as opt:
        dset = opt.create_dataset('Answer', data=Result, compression='gzip')
        dset.attrs['Method'] = method
        print('The output file path is {}'.format(optpath))


GainTable = np.loadtxt(gaintablefile, skiprows=0, usecols=2)
N = 3
SLICES = np.append(np.arange(0, len(filenos), N), len(filenos))
tic = time.time()
cpu_tic = time.process_time()
with Pool(N) as pool :
    Waveforms_and_info = pd.concat(pool.map(Read_Data, filenos[SLICES[0]:SLICES[1]]))
print('Data Loaded, consuming {0:.4f}s using {1} threads, cpu time {2:.4f}s'.format(time.time() - tic, N, time.process_time() - cpu_tic))
for i in range(len(SLICES) - 1):
    FILENOS = filenos[SLICES[i]:SLICES[i + 1]]
    with Pool(N) as pool_read, Pool(N) as pool_write :
        if i < (len(SLICES) - 2) :
            NEXT_FILENOS = filenos[SLICES[i + 1]:SLICES[i + 2]]
            WaveReader = pool_read.map_async(Read_Data, NEXT_FILENOS)

        WindowSize = len(Waveforms_and_info["Waveform"][0:1][0])
        Channel_Grouped_Waveform = Waveforms_and_info.groupby(by='ChannelID')

        filter_limit = 0.05
        Timeline = np.arange(WindowSize, dtype=np.int16).reshape(1, WindowSize)

        def Forward(channelid) :
            SPECharge = GainTable[channelid]
            filter_limit = 0.01 * SPECharge
            Data_of_this_channel = Channel_Grouped_Waveform.get_group(channelid)
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
        Grouped_Result = Result.groupby(by='FileNo')
        print('Prediction generated, real time {0:.4f}s, cpu time {1:.4f}s'.format(time.time() - tic, time.process_time() - cpu_tic))

        tic = time.time()
        cpu_tic = time.process_time()
        pool_write.starmap(WriteData, [(fn, Grouped_Result) for fn in FILENOS])
        print('output written, real time {0:.4f}s, cpu time {1:.4f}s'.format(time.time() - tic, time.process_time() - cpu_tic))

        if i < (len(SLICES) - 1) :
            tic = time.time()
            cpu_tic = time.process_time()
            WaveReader.wait()
            Waveforms_and_info = pd.concat(WaveReader.get())
            print('reading the next inputs, real time {0:.4f}s, cpu time {1:.4f}s'.format(time.time() - tic, time.process_time() - cpu_tic))


print('Finished! Consuming {0:.2f}s in total, cpu time {1:.2f}s.'.format(time.time() - global_start, time.process_time() - cpu_global_start))
