import os
import random
import argparse
from multiprocessing import Pool, cpu_count
from time import time
global_start = time()

import numpy as np
import tables
import h5py
import numba
import pandas as pd

import wf_func as wff

psr = argparse.ArgumentParser()
psr.add_argument('ipt', help='input file direction & prefix')
psr.add_argument('-o', '--outputdir', dest='opt', help='output_dir')
psr.add_argument('--ref', type=str, nargs='+', help='reference file')
args = psr.parse_args()
SavePath = args.opt
reference = args.ref
filedir, prefix = os.path.split(args.ipt)
files = os.listdir(filedir)
files = [filedir+'/'+fi for fi in files if prefix in fi and not os.path.isdir(fi) and '.h5' == fi[-3:]]

def Make_Time_Vector(GroundTruth, Waveforms_and_info, mode) :
    i = 0
    Wave_TriggerNo = Waveforms_and_info['TriggerNo'].to_numpy()
    Truth_TriggerNo = GroundTruth['TriggerNo'].to_numpy(); nt = len(Truth_TriggerNo)
    HitPosInWindow = np.around(GroundTruth['HitPosInWindow'].to_numpy()).astype(np.int)
    if mode == 'Charge':
        Time_Series = np.zeros((len(Waveforms_and_info), WindowSize), dtype=np.float64)
        Mode = GroundTruth[mode].to_numpy()
    elif mode == 'PEnum':
        Time_Series = np.zeros((len(Waveforms_and_info), WindowSize), dtype=np.uint8)
        Mode = np.ones(len(HitPosInWindow), dtype=np.float64)
    for j in range(len(Waveforms_and_info)) :
        while i < nt and Wave_TriggerNo[j] == Truth_TriggerNo[i] :
            Time_Series[j][HitPosInWindow[i]] = Time_Series[j][HitPosInWindow[i]] + Mode[i]
            i = i + 1
    return Time_Series

def TableToDataFrame(Array) :
    return(pd.DataFrame({name: list(Array[name]) for name in Array.dtype.names}))

def Read_Data(sliceNo, filename, Wave_startentry, Wave_endentry, Truth_startentry, Truth_endentry) :
    h5file = tables.open_file(filename, 'r')
    WaveformTable = h5file.root.Readout.Waveform
    GroundTruthTable = h5file.root.SimTriggerInfo.PEList
    print('Reading File ' + filename)
    Waveforms_and_info = WaveformTable[Wave_startentry:Wave_endentry]
    GroundTruth = GroundTruthTable[Truth_startentry:Truth_endentry]
    GroundTruth = GroundTruth[np.logical_and(GroundTruth['HitPosInWindow'] >= 0, GroundTruth['HitPosInWindow'] < WindowSize)]
    h5file.close()
    return (sliceNo, {'Waveform': TableToDataFrame(Waveforms_and_info), 'GroundTruth': TableToDataFrame(GroundTruth)})

def PreProcess(channelid) :
    print('PreProcessing channel {:02d}'.format(channelid))
    Waves_of_this_channel = Grouped_Waves.get_group(channelid)
    Truth_of_this_channel = Grouped_Truth.get_group(channelid)
    Origin_Waves = np.vstack(Waves_of_this_channel['Waveform'].to_numpy())
    Shifted_Waves = np.empty((len(Origin_Waves), WindowSize), dtype=np.float32)
    for i in range(len(Origin_Waves)) :
        Shifted_Waves[i] = Origin_Waves[i].astype(np.float64) * spe_pre[channelid]['epulse']

    PEnumSpectrum = Make_Time_Vector(Truth_of_this_channel, Waves_of_this_channel, 'PEnum')
    ChargeSpectrum = Make_Time_Vector(Truth_of_this_channel, Waves_of_this_channel, 'Charge')
    with h5py.File(SavePath + '{:02d}.h5'.format(channelid), 'w') as opt:
        opt.create_dataset('Waveform', data=Shifted_Waves, compression='gzip')
        opt.create_dataset('PEnumSpectrum', data=PEnumSpectrum, compression='gzip')
        opt.create_dataset('ChargeSpectrum', data=ChargeSpectrum, compression='gzip')
    return

print('training data pre-processing savepath is {}'.format(os.path.dirname(SavePath)))
spe_pre = wff.read_model(reference[0])
sliceNo = 0
trainfile_list = []  # index sliceNo; value: number of waveforms to be readed
start = time()
h5file = tables.open_file(files[0], 'r')
twfnum = len(h5file.root.Readout.Waveform)
h5file.close()
random.seed(0)
files = random.sample(files, min(int(1e6/twfnum), len(files)))
files.sort()
print('Used files are: ', files)
for filename in files :
    h5file = tables.open_file(filename, 'r')
    if filename == files[0]:
        WindowSize = len(h5file.root.Readout.Waveform[0]['Waveform'])
    Waveform_Len = len(h5file.root.Readout.Waveform)
    GroundTruth_Len = len(h5file.root.SimTriggerInfo.PEList)
    slices = np.append(np.arange(0, Waveform_Len, 10000), Waveform_Len)
    Truth_slices = (GroundTruth_Len / Waveform_Len * slices).astype(np.int)
    for i in range(len(slices) - 1) :
        trainfile_list.append((sliceNo, filename, slices[i], slices[i + 1], Truth_slices[i], Truth_slices[i + 1]))
        sliceNo = sliceNo + 1
    h5file.close()
print('Initialization Finished, consuming {:.02f}s.'.format(time() - start))
with Pool(min(len(trainfile_list), cpu_count())) as pool :
    Reading_Result = dict(pool.starmap(Read_Data, trainfile_list))
Waveforms_and_info = pd.concat([Reading_Result[i]['Waveform'] for i in range(len(trainfile_list))])
GroundTruth = pd.concat([Reading_Result[i]['GroundTruth'] for i in range(len(trainfile_list))])
print('Data Loaded, consuming {:.02f}s'.format(time() - start))

Grouped_Waves = Waveforms_and_info.groupby(by='ChannelID')
Grouped_Truth = GroundTruth.groupby(by='PMTId')

channelid_list = np.unique(Waveforms_and_info['ChannelID'].to_numpy())

start = time()
with Pool(min(len(channelid_list), cpu_count())) as pool :
    pool.map(PreProcess, channelid_list)
print('Data Saved, consuming {:.02f}s'.format(time() - start))
print('Finished, consuming {:.02f}s in total.'.format(time() - global_start))