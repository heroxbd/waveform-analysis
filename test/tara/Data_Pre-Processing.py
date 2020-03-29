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


import os
import time

import numpy as np
import tables

from JPwaptool_Lite import JPwaptool_Lite


def AssignStream(WindowSize) :
    if WindowSize >= 1000 :
        stream = JPwaptool_Lite(WindowSize, 100, 600)
    elif WindowSize == 600 :
        stream = JPwaptool_Lite(WindowSize, 50, 400)
    else:
        raise ValueError("Unknown WindowSize, I don't know how to choose the parameters for pedestal calculatation")
    return stream


time_start = time.time()


# Make Directory
if not os.path.exists(SavePath):
    os.makedirs(SavePath)
print("training data pre-processing savepath is {}".format(SavePath))

# Read hdf5 file
FileNo = 0
Len_Entry = 0
streams = []

while(Len_Entry < Nwav) :
    try :
        h5file = tables.open_file(prefix + "{}".format(FileNo) + ".h5", "r")
    except OSError :
        raise ValueError("You demand too many training waves while providing insufficient input files!\nCheck if your taining files have continuous file numbers starting with 0.")
    # Reading file
    WaveformTable = h5file.root.Waveform
    if len(WaveformTable) < Nwav :
        Len_Entry = len(WaveformTable)
    else :
        Len_Entry = Nwav
    Nwav = Nwav - Len_Entry
    Waveforms_and_info = WaveformTable[0:Len_Entry]

    # Global Initialization
    if FileNo == 0 :
        example_wave = Waveforms_and_info[0:100]["Waveform"]
        is_positive_pulse = (example_wave.max(axis=1) - example_wave.mean(axis=1)) > (example_wave.mean(axis=1) - example_wave.min(axis=1))
        if sum(is_positive_pulse) > 95 :  # positive pulse
            is_positive_pulse = True
        elif sum(is_positive_pulse) > 5 :
            raise ValueError("ambiguous pulse!")
        else :
            is_positive_pulse = False

        WindowSize = len(example_wave[0])

        class PreProcessedData(tables.IsDescription):
            PET = tables.Col.from_type('float32', shape=WindowSize, pos=0)
            Wave = tables.Col.from_type('float32', shape=WindowSize, pos=1)
        ChannelIDs = set(Waveforms_and_info["ChannelID"])
        Prefile = dict([])
        TrainDataTable = dict([])
        traindata = dict([])
        for ChannelID in ChannelIDs :
            # create Pre-Processed output file
            Prefile[ChannelID] = tables.open_file(SavePath + "Pre_Channel{}.h5".format(ChannelID), mode="w", title="Pre-Processed-Training-Data")
            # Create group and tables
            TrainDataTable[ChannelID] = Prefile[ChannelID].create_table("/", "TrainDataTable", PreProcessedData, "Wave and PET")
            traindata[ChannelID] = TrainDataTable[ChannelID].row
        entry = 0

    # File Initialize
    if(is_positive_pulse) : Waveforms_and_info["Waveform"] = Waveforms_and_info["Waveform"].max() - Waveforms_and_info["Waveform"]
    GroundTruthTable = h5file.root.GroundTruth
    GroundTruth_Len = min(round(len(GroundTruthTable) / len(WaveformTable) * Len_Entry * 2), len(GroundTruthTable))
    last_eventid = Waveforms_and_info["EventID"][-1]
    last_channelid = Waveforms_and_info["ChannelID"][-1]
    GroundTruth = GroundTruthTable[0:GroundTruth_Len]
    streams.append(AssignStream(WindowSize))  # re-asign stream to avoid bus error
    if len(streams) > 1 : del streams[-2]
    TimeSeries = streams[-1].Make_Time_Vector(GroundTruth["EventID"], GroundTruth["ChannelID"], GroundTruth["PETime"], np.int64(last_eventid), np.int16(last_channelid), np.int64(Len_Entry))

    streams.append(AssignStream(WindowSize))  # re-asign stream to avoid bus error
    del streams[-2]
    print("stream reassigned")
    for i, w in enumerate(Waveforms_and_info) :  # loop all events in this file
        # check point
        if (entry) % 100000 == 0:
            print("Currently in file {0} entry {1}, processed {2} entries, progress {3:.2f}%".format(FileNo, i, entry, entry / args.Nwav * 100))
        streams[-1].Calculate(w["Waveform"])
        traindata[w["ChannelID"]]['Wave'] = streams[-1].ChannelInfo.Ped - w["Waveform"]
        traindata[w["ChannelID"]]['PET'] = TimeSeries[i]
        traindata[w["ChannelID"]].append()
        entry = entry + 1

    for TD in list(TrainDataTable.values()) :
        TD.flush()

    FileNo = FileNo + 1

h5file.close()
for Pf in list(Prefile.values()) :
    Pf.close()
time_end = time.time()
print('consuming time: {}s'.format(time_end - time_start))
