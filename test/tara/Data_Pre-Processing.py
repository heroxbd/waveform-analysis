#!/usr/bin/env python3
import argparse
psr = argparse.ArgumentParser()
psr.add_argument('ipt', help='input file prefix')
psr.add_argument('-o', '--outputdir', dest='opt', help='output_dir')
psr.add_argument('-N', '--NWaves', dest='Nwav', type=int, help='entries of waves')
psr.add_argument('-f', '--format', dest='format', default='h5', choices=['h5', 'root'], help='input file format')
args = psr.parse_args()
Nwav = args.Nwav
prefix = args.ipt
suffix = args.format
SavePath = args.opt
if(Nwav <= 100):
    raise ValueError("NWaves must > 100 !")

from time import time
start = time()
global_start = time()

import os
import numpy as np
import tables
import numba
from multiprocessing import Pool
from JPwaptool_Lite import JPwaptool_Lite
import pandas as pd
from tqdm import tqdm


@numba.jit
def Make_Time_Vector(GroundTruth, Waveforms_and_info) :
    i = 0
    Time_Series = np.zeros((len(Waveforms_and_info), WindowSize), dtype=np.float32)
    Wave_EventID = Waveforms_and_info["EventID"].to_numpy()
    Truth_EventID = GroundTruth["EventID"].to_numpy()
    PETime = GroundTruth["PETime"].to_numpy()
    for j in range(len(Waveforms_and_info)) :
        while(Wave_EventID[j] == Truth_EventID[i]) :
            Time_Series[j][PETime[i]] = Time_Series[j][PETime[i]] + 1
            i = i + 1
    return Time_Series


def AssignStream(WindowSize) :
    if WindowSize >= 1000 :
        stream = JPwaptool_Lite(WindowSize, 100, 600)
    elif WindowSize == 600 :
        stream = JPwaptool_Lite(WindowSize, 50, 400)
    else:
        raise ValueError("Unknown WindowSize, I don't know how to choose the parameters for pedestal calculatation")
    return stream


# Make Directory
if not os.path.exists(SavePath):
    os.makedirs(SavePath)
print("training data pre-processing savepath is {}".format(SavePath))
print("Initialization Finished, consuming {:.5f}s.".format(time() - start))


# Read hdf5 file
def TableToDataFrame(Array) :
    return(pd.DataFrame({name: list(Array[name]) for name in Array.dtype.names}))


def Read_Data(sliceNo, filename, Wave_startentry, Wave_endentry, Truth_startentry, Truth_endentry) :
    h5file = tables.open_file(filename, "r")
    WaveformTable = h5file.root.Waveform
    GroundTruthTable = h5file.root.GroundTruth
    print("Reading File " + filename)
    Waveforms_and_info = WaveformTable[Wave_startentry:Wave_endentry]
    GroundTruth = GroundTruthTable[Truth_startentry:Truth_endentry]
    h5file.close()

    example_wave = Waveforms_and_info[0:100]["Waveform"]
    is_positive_pulse = (example_wave.max(axis=1) - example_wave.mean(axis=1)) > (example_wave.mean(axis=1) - example_wave.min(axis=1))
    if sum(is_positive_pulse) > 95 :  # positive pulse
        is_positive_pulse = True
    elif sum(is_positive_pulse) > 5 :
        raise ValueError("ambiguous pulse!")
    else :
        is_positive_pulse = False
    if(is_positive_pulse) : Waveforms_and_info["Waveform"] = Waveforms_and_info["Waveform"].max() - Waveforms_and_info["Waveform"]
    return (sliceNo, {"Waveform": TableToDataFrame(Waveforms_and_info), "GroundTruth": TableToDataFrame(GroundTruth)})


FileNo = 0
sliceNo = 0
trainfile_list = []  # index sliceNo; value: number of waveforms to be readed
start = time()
while(Nwav != 0) :
    filename = prefix + "{}".format(FileNo) + ".h5"
    try :
        h5file = tables.open_file(filename, "r")
    except OSError :
        raise ValueError("You demand too many training waves while providing insufficient input files!\nCheck if your taining files have continuous file numbers starting with 0.")
    # Reading file
    if FileNo == 0 :
        origin_dtype = h5file.root.Waveform.dtype
        WindowSize = origin_dtype["Waveform"].shape[0]
        gpufloat_dtype = np.dtype([(name, np.dtype('float32') if name == "Waveform" else origin_dtype[name].base, origin_dtype[name].shape) for name in origin_dtype.names])
    Waveform_Len = len(h5file.root.Waveform)
    GroundTruth_Len = len(h5file.root.GroundTruth)
    if Waveform_Len > Nwav :
        Waveform_Len = Nwav
        GroundTruth_Len = min(round(GroundTruth_Len / len(h5file.root.Waveform) * Waveform_Len * 2), GroundTruth_Len)
    elif Waveform_Len == Nwav :  # in case that the end of the last file is read, resulting in Index out of range when calling Make_Time_Vector
        Nwav = Nwav - 1
        Waveform_Len = Nwav
    Nwav = Nwav - Waveform_Len
    slices = np.append(np.arange(0, Waveform_Len, 150000), Waveform_Len)
    Truth_slices = (GroundTruth_Len / Waveform_Len * slices).astype(np.int)
    for i in range(len(slices) - 1) :
        trainfile_list.append((sliceNo, filename, slices[i], slices[i + 1], Truth_slices[i], Truth_slices[i + 1]))
        sliceNo = sliceNo + 1
    FileNo = FileNo + 1
    h5file.close()
with Pool(len(trainfile_list)) as pool :
    Reading_Result = dict(pool.starmap(Read_Data, trainfile_list))
Waveforms_and_info = pd.concat([Reading_Result[i]["Waveform"] for i in range(len(trainfile_list))])
GroundTruth = pd.concat([Reading_Result[i]["GroundTruth"] for i in range(len(trainfile_list))])
print("Data Loaded, consuming {:.5f}s".format(time() - start))


Grouped_Waves = Waveforms_and_info.groupby(by="ChannelID")
Grouped_Truth = GroundTruth.groupby(by="ChannelID")


class PreProcessedData(tables.IsDescription):
    HitSpectrum = tables.Col.from_type('float32', shape=WindowSize, pos=0)
    Waveform = tables.Col.from_type('float32', shape=WindowSize, pos=1)


def PreProcess(channelid, Waves_of_this_channel, Truth_of_this_channel) :
    print("PreProcessing channel {}".format(channelid))
    Origin_Waves = Waves_of_this_channel["Waveform"].to_numpy()
    Shifted_Waves = np.empty((len(Origin_Waves), WindowSize), dtype=np.float32)
    HitSpectrum = np.empty((len(Origin_Waves), WindowSize), dtype=np.float32)

    stream = AssignStream(WindowSize)
    for i, w in enumerate(Waves_of_this_channel["Waveform"]) :
        stream.Calculate(w)
        Shifted_Waves[i] = stream.ChannelInfo.Ped - w

    HitSpectrum = Make_Time_Vector(Truth_of_this_channel, Waves_of_this_channel)

    # create Pre-Processed output file
    Prefile = tables.open_file(SavePath + "/Pre_Channel{}.h5".format(channelid), mode="w", title="Pre-Processed-Training-Data")
    # Create group and tables
    TrainDataTable = Prefile.create_table("/", "TrainDataTable", PreProcessedData, "Wave and HitSpectrum")
    TrainDataTable.append([list(HitSpectrum), list(Shifted_Waves)])
    TrainDataTable.flush()
    Prefile.close()


channelid_list = np.unique(Waveforms_and_info["ChannelID"].to_numpy())

start = time()
PreProcess_Parameters = []
for channelid in tqdm(channelid_list, desc="Grouping channels") :
    Waves_of_this_channel = Grouped_Waves.get_group(channelid).reset_index(drop=True, inplace=False)
    Truth_of_this_channel = Grouped_Truth.get_group(channelid).reset_index(drop=True, inplace=False)
    PreProcess_Parameters.append((channelid, Waves_of_this_channel , Truth_of_this_channel))
with Pool(len(channelid_list)) as pool :
    pool.starmap(PreProcess, PreProcess_Parameters)
print("Data Saved, consuming {:.5f}s".format(time() - start))
print("Finished, consuming {:.4f}s in total.".format(time() - global_start))
