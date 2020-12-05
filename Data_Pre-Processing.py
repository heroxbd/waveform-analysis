#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from multiprocessing import Pool, cpu_count
from time import time
global_start = time()

import numpy as np
import tables
import h5py
import pandas as pd
import uproot4
import awkward1 as ak
from IPython import embed

psr = argparse.ArgumentParser()
psr.add_argument('ipt', help='input rawdata file')
psr.add_argument('bsl', help='input baseline file')
psr.add_argument('-o', '--outputdir', dest='opt', help='output_dir')
args = psr.parse_args()
SavePath = args.opt
rawfilename = args.ipt
bslfilename = args.bsl

with tables.open_file(rawfilename, 'r') as h5file :
    WindowSize = len(h5file.root.Readout.Waveform[0]["Waveform"])


def Make_Time_Vector(GroundTruth, Waveforms_and_info, mode) :
    GroundTruth[GroundTruth['Charge'] > 0]
    i = 0
    Wave_TriggerNo = Waveforms_and_info['TriggerNo'].to_numpy()
    Truth_TriggerNo = GroundTruth['EventID'].to_numpy()
    nt = len(Truth_TriggerNo)
    HitPosInWindow = GroundTruth['RiseTime'].to_numpy()
    if mode == 'Charge':
        Time_Series = np.zeros((len(Waveforms_and_info), WindowSize), dtype=np.float64)
        Mode = GroundTruth[mode].to_numpy()
    elif mode == 'PEnum':
        Time_Series = np.zeros((len(Waveforms_and_info), WindowSize), dtype=np.uint8)
        Mode = np.ones(len(HitPosInWindow), dtype=np.float64)
    for j in range(len(Waveforms_and_info)) :
        while i < nt and Wave_TriggerNo[j] == Truth_TriggerNo[i] :
            Time_Series[j][int(round(HitPosInWindow[i]))] = Time_Series[j][int(round(HitPosInWindow[i]))] + max(Mode[i], 0)
            i = i + 1
    return Time_Series


def TableToDataFrame(Array) :
    return(pd.DataFrame({name: list(Array[name]) for name in Array.dtype.names}))


def Read_Data(filename) :
    h5file = tables.open_file(filename, 'r')
    WaveformTable = h5file.root.Readout.Waveform
    GroundTruthTable = h5file.root.SimTriggerInfo.PEList
    print('Reading File ' + filename)
    Waveforms_and_info = WaveformTable[:]
    GroundTruth = GroundTruthTable[:]
    GroundTruth = GroundTruth[np.logical_and(GroundTruth['RiseTime'] >= 0, GroundTruth['RiseTime'] < WindowSize)]
    h5file.close()
    return {'Waveform': TableToDataFrame(Waveforms_and_info), 'GroundTruth': TableToDataFrame(GroundTruth)}


def ReadBaseline(filename) :
    f = uproot4.open(filename)
    Ped = ak.flatten(f["SimpleAnalysis"]["ChannelInfo.Pedestal"].array())
    CID = ak.flatten(f["SimpleAnalysis"]["ChannelInfo.ChannelId"].array())
    Cha = ak.flatten(f["SimpleAnalysis"]["ChannelInfo.Charge"].array())
    return pd.DataFrame({"ChannelID": CID, "Pedestal": Ped, "Charge": Cha})


def PreProcess(channelid) :
    print('PreProcessing channel {:02d}'.format(channelid))
    Waves_of_this_channel = Grouped_Waves.get_group(channelid)
    Truth_of_this_channel = Grouped_Truth.get_group(channelid)
    Bslne_of_this_channel = Grouped_Bslne.get_group(channelid)
    Baselines = np.vstack(Bslne_of_this_channel['Pedestal'].to_numpy())
    Origin_Waves = np.vstack(Waves_of_this_channel['Waveform'].to_numpy())
    Shifted_Waves = np.empty((len(Origin_Waves), WindowSize), dtype=np.float32)
    for i in range(len(Origin_Waves)) :
        Shifted_Waves[i] = Baselines[i] - Origin_Waves[i]

    PEnumSpectrum = Make_Time_Vector(Truth_of_this_channel, Waves_of_this_channel, 'PEnum')
    ChargeSpectrum = Make_Time_Vector(Truth_of_this_channel, Waves_of_this_channel, 'Charge')
    with h5py.File(SavePath + '{:02d}.h5'.format(channelid), 'w') as opt:
        opt.create_dataset('Waveform', data=Shifted_Waves, compression='gzip')
        opt.create_dataset('PEnumSpectrum', data=PEnumSpectrum, compression='gzip')
        opt.create_dataset('ChargeSpectrum', data=ChargeSpectrum, compression='gzip')
    return


print('training data pre-processing savepath is {}'.format(os.path.dirname(SavePath)))
sliceNo = 0
trainfile_list = []  # index sliceNo; value: number of waveforms to be readed
start = time()
print('Initialization Finished, consuming {:.5f}s.'.format(time() - start))
Reading_Result = Read_Data(rawfilename)
Waveforms_and_info = Reading_Result["Waveform"]
GroundTruth = Reading_Result['GroundTruth']
Baseline = ReadBaseline(bslfilename)
print('Data Loaded, consuming {:.5f}s'.format(time() - start))

Grouped_Waves = Waveforms_and_info.groupby(by='ChannelID')
Grouped_Truth = GroundTruth.groupby(by='ChannelID')
Grouped_Bslne = Baseline.groupby(by='ChannelID')

channelid_list = np.unique(Waveforms_and_info['ChannelID'].to_numpy())

start = time()
with Pool(min(len(channelid_list), cpu_count())) as pool :
    pool.map(PreProcess, channelid_list)
print('Data Saved, consuming {:.5f}s'.format(time() - start))
print('Finished, consuming {:.4f}s in total.'.format(time() - global_start))
