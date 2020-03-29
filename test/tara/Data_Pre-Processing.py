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

#from IPython import embed
from time import time
start = time()
time_start = time()

import os
import numpy as np
import tables
import numba

from JPwaptool_Lite import JPwaptool_Lite


@numba.jit
def Make_Time_Vector(EventIDs, ChannelIDs, PETimes, WindowSize, nWaves) :
    last_eventid = EventIDs[0]
    last_channelid = ChannelIDs[0]
    i = 0
    j = 0
    Time_Series = np.zeros((nWaves, WindowSize), dtype=np.float)
    while(j != nWaves) :
        Time_Series[j][PETimes[i]] = Time_Series[j][PETimes[i]] + 1
        i = i + 1
        if last_eventid != EventIDs[i] or last_channelid != ChannelIDs[i] :
            j = j + 1
            last_eventid = EventIDs[i]
            last_channelid = ChannelIDs[i]
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
FileNo = 0
Len_Entry = 0
start = time()
while(Nwav != 0) :
    try :
        h5file = tables.open_file(prefix + "{}".format(FileNo) + ".h5", "r")
    except OSError :
        raise ValueError("You demand too many training waves while providing insufficient input files!\nCheck if your taining files have continuous file numbers starting with 0.")
    # Reading file
    WaveformTable = h5file.root.Waveform
    GroundTruthTable = h5file.root.GroundTruth
    if len(WaveformTable) < Nwav :
        Len_Entry = len(WaveformTable)
    else :
        Len_Entry = Nwav
    Nwav = Nwav - Len_Entry
    GroundTruth_Len = min(round(len(GroundTruthTable) / len(WaveformTable) * Len_Entry * 2), len(GroundTruthTable))
    print("Reading File {}".format(FileNo))
    if FileNo == 0 :
        Waveforms_and_info = WaveformTable[0:Len_Entry]
        GroundTruth = GroundTruthTable[0:GroundTruth_Len]
    else :
        Waveforms_and_info = np.hstack((Waveforms_and_info, WaveformTable[0:Len_Entry]))
        GroundTruth = np.hstack((GroundTruth, GroundTruthTable[0:GroundTruth_Len]))
    FileNo = FileNo + 1
print("Data Loaded, consuming {:.5f}s".format(time() - start))

# Global Initialization
start = time()
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

if(is_positive_pulse) : Waveforms_and_info["Waveform"] = Waveforms_and_info["Waveform"].max() - Waveforms_and_info["Waveform"]

print("Parameter Configures, consuming {:.5f}s".format(time() - start))

start = time()
TimeSeries = Make_Time_Vector(GroundTruth["EventID"], GroundTruth["ChannelID"], GroundTruth["PETime"], WindowSize, len(Waveforms_and_info))
print("TimeSeries Made, consuming {:.5f}s".format(time() - start))

entry = 0
stream = AssignStream(WindowSize)
start = time()
for i, w in enumerate(Waveforms_and_info) :  # loop all events in this file
    # check point
    if (entry) % 100000 == 0:
        print("processed {0} entries, progress {1:.2f}%".format(entry, entry / args.Nwav * 100))
    stream.Calculate(w["Waveform"])
    traindata[w["ChannelID"]]['Wave'] = stream.ChannelInfo.Ped - w["Waveform"]
    traindata[w["ChannelID"]]['PET'] = TimeSeries[i]
    traindata[w["ChannelID"]].append()
    entry = entry + 1

print("Process Done, consuming {:.5f}s".format(time() - start))
start = time()

for TD in list(TrainDataTable.values()) :
    TD.flush()

h5file.close()
for Pf in list(Prefile.values()) :
    Pf.close()
print("Data Saved, consuming {:.5f}s".format(time() - start))
print('consuming time: {}s'.format(time() - time_start))
