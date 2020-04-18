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

# from IPython import embed
from time import time
start = time()
time_start = time()

import os
import numpy as np
import tables
import numba
from multiprocessing import Pool
from JPwaptool_Lite import JPwaptool_Lite
import pandas as pd
from IPython import embed


@numba.jit
def Make_Time_Vector(GroundTruth, Waveforms) :
    i = 0
    Time_Series = np.zeros(Waveforms["Waveform"].shape, dtype=np.float32)
    for j in range(len(Waveforms)) :
        while(Waveforms["EventID"][j] == GroundTruth["EventID"][i] and (Waveforms["ChannelID"][j] == GroundTruth["ChannelID"][i])) :
            Time_Series[j][GroundTruth["PETime"][i]] = Time_Series[j][GroundTruth["PETime"][i]] + 1
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
    Shifted_Waves_and_info = np.empty(Waveforms_and_info.shape, dtype=gpufloat_dtype)
    for name in origin_dtype.names :
        if name != "Waveform" :
            Shifted_Waves_and_info[name] = Waveforms_and_info[name]
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

    stream = AssignStream(WindowSize)
    for i, w in enumerate(Waveforms_and_info["Waveform"]) :
        stream.Calculate(w)
        Shifted_Waves_and_info[i]["Waveform"] = stream.ChannelInfo.Ped - w
    return (sliceNo, {"Waveform": TableToDataFrame(Shifted_Waves_and_info), "GroundTruth": TableToDataFrame(GroundTruth)})


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
    Nwav = Nwav - Waveform_Len
    slices = np.append(np.arange(0, Waveform_Len, 150000), Waveform_Len)
    Truth_slices = (GroundTruth_Len / Waveform_Len * slices).astype(np.int)
    print(slices)
    print(Truth_slices)
    for i in range(len(slices) - 1) :
        trainfile_list.append((sliceNo, filename, slices[i], slices[i + 1], Truth_slices[i], Truth_slices[i + 1]))
        sliceNo = sliceNo + 1
    FileNo = FileNo + 1
    h5file.close()
# print(list(trainfile_set.items()))
with Pool(int(np.ceil(len(trainfile_list) / 2))) as pool :
    Reading_Result = pool.starmap(Read_Data, trainfile_list)
print("Data Loaded, consuming {:.5f}s".format(time() - start))


#ChannelIDs = set(Waveforms_and_info["ChannelID"])
#Prefile = dict([])
#TrainDataTable = dict([])
#traindata = dict([])
#for ChannelID in ChannelIDs :
#    # create Pre-Processed output file
#    Prefile[ChannelID] = tables.open_file(SavePath + "/Pre_Channel{}.h5".format(ChannelID), mode="w", title="Pre-Processed-Training-Data")
#    # Create group and tables
#    TrainDataTable[ChannelID] = Prefile[ChannelID].create_table("/", "TrainDataTable", PreProcessedData, "Wave and PET")
#    traindata[ChannelID] = TrainDataTable[ChannelID].row
#
#
#print("Parameter Configures, consuming {:.5f}s".format(time() - start))
#
#start = time()
#TimeSeries = Make_Time_Vector(GroundTruth, Waveforms_and_info)
#print("TimeSeries Made, consuming {:.5f}s".format(time() - start))
#
#start = time()
#for i, w in enumerate(Waveforms_and_info) :  # loop all events in this file
#    # check point
#    if (entry) % 100000 == 0:
#        print("processed {0} entries, progress {1:.2f}%".format(entry, entry / args.Nwav * 100))
#    stream.Calculate(w["Waveform"])
#    traindata[w["ChannelID"]]['Wave'] = stream.ChannelInfo.Ped - w["Waveform"]
#    traindata[w["ChannelID"]]['PET'] = TimeSeries[i]
#    traindata[w["ChannelID"]].append()
#    entry = entry + 1
#
#
#class PreProcessedData(tables.IsDescription):
#    HitSpectrum = tables.Col.from_type('float32', shape=WindowSize, pos=0)
#    Waveform = tables.Col.from_type('float32', shape=WindowSize, pos=1)
#
#
#print("Process Done, consuming {:.5f}s".format(time() - start))
#start = time()
#
#for TD in list(TrainDataTable.values()) :
#    TD.flush()
#
#for Pf in list(Prefile.values()) :
#    Pf.close()
#print("Data Saved, consuming {:.5f}s".format(time() - start))
#print('consuming time: {}s'.format(time() - time_start))
