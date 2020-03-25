import time
time_start = time.time()
import numpy as np
import os
import sys
import tables
from JPwaptool_Lite import JPwaptool_Lite

## Initialization

# Make Directory
SavePath = sys.argv[2]
if not os.path.exists(SavePath):
    os.makedirs(SavePath)
print("training data pre-processing savepath is {}".format(SavePath))

# Read hdf5 file

fullfilename = sys.argv[1]
FileName = os.path.basename(fullfilename).replace(".h5",'')
h5file = tables.open_file(fullfilename, "r")
WaveformTable = h5file.root.Waveform
GroundTruthTable = h5file.root.GroundTruth
Len_Entry = len(WaveformTable)

max_set_number = int(sys.argv[3])
if max_set_number>0 :
    Len_Entry = min(max_set_number,Len_Entry)

if(Len_Entry<=100) : raise ValueError("max_set_number must > 100 !")

print(Len_Entry, "data entries") # Entry 10^6

# Make Data Matrix
Waveforms_and_info = WaveformTable[0:Len_Entry]
example_wave = Waveforms_and_info["Waveform"][0:100]
WindowSize = len(example_wave[0])
is_positive_pulse = (example_wave.max(axis=1)-example_wave.mean(axis=1))>(example_wave.mean(axis=1)-example_wave.min(axis=1))
Waveforms = Waveforms_and_info["Waveform"]
if sum(is_positive_pulse)>95 : #positive pulse
    Waveforms = Waveforms.max()-Waveforms
elif sum(is_positive_pulse)>5 : raise ValueError("ambiguous pulse!")

if WindowSize>=1000 :
    stream = JPwaptool_Lite(WindowSize,100,600)
elif WindowSize==600 :
    stream = JPwaptool_Lite(WindowSize,50,400)
else : 
    raise ValueError("Unknown WindowSize, I don't know how to choose the parameters for pedestal calculatation")

class PreProcessedData(tables.IsDescription):
    PET = tables.Col.from_type('float32', shape=WindowSize, pos=0)
    Wave = tables.Col.from_type('float32', shape=WindowSize, pos=1)
    
#create Pre-Processed output file
Prefile = tables.open_file(SavePath+"Pre.h5", mode="w", title="Pre-Processed-Training-Data")

# Create group and tables
group = "/"
TrainDataTable = Prefile.create_table(group, "TrainDataTable", PreProcessedData, "Wave and PET")
traindata = TrainDataTable.row

GroundTruth_Len = min(round(len(GroundTruthTable)/len(WaveformTable)*Len_Entry*2),len(GroundTruthTable))
last_eventid = Waveforms_and_info["EventID"][-1]
last_channelid = Waveforms_and_info["ChannelID"][-1]
GroundTruth = GroundTruthTable[0:GroundTruth_Len]
TimeSeries = stream.Make_Time_Vector(GroundTruth["EventID"],GroundTruth["ChannelID"],GroundTruth["PETime"],np.int64(last_eventid),np.int16(last_channelid),np.int64(Len_Entry))

for entry in range(Len_Entry):
    stream.Calculate(Waveforms[entry])
    traindata['Wave'] = stream.ChannelInfo.Ped - Waveforms[entry] 
    traindata['PET'] = TimeSeries[entry]
    traindata.append()

    # check point
    if (entry+1)%5000==0:
        print(entry+1)
TrainDataTable.flush()
h5file.close()
Prefile.close()
time_end = time.time()
print('consuming time: {}s'.format(time_end-time_start))
