import tables
import numpy as np
import os
import sys

## Initialization
fileBriefname = sys.argv[2]
# Make Directory
SavePath = "./{}/Prediction_Pre-Processing_Results/".format(fileBriefname)
if not os.path.exists(SavePath):
    os.makedirs(SavePath)


# Read hdf5 file
'''
LoadPath= "./DataSets/"
FileName = "zincm-problem"
'''
LoadPath = '/home/greatofdream/killxbq/'
# FileName = 'training'
# FileName = 'ztraining-1'
# filename = LoadPath+FileName+".h5"
filename = sys.argv[1]

h5file = tables.open_file(filename, "r")
WaveformTable = h5file.root.Waveform
Len_Entry = len(WaveformTable)
print(Len_Entry, "data entries") # Entry 10^6
# WaveChannel 0-1028, length 1029

# Make Data Matrix
def make_wave_long_vec(wave_form):
    # non_negative_peak + zero_base_level
    shift= np.argmax(np.bincount(wave_form)) #make baseline shifts, normally 972
    shift_wave=-wave_form+shift #non_negative_peak + zero_base_level
    return shift_wave

# Data Pre-Processing
Num_Entry = Len_Entry #100000 #Len_Entry
EventMat=[]
ChanMat=[]
WaveMat=[]
save_period = 200000

#saving params
start_entry = 0
entry_index = 0

for entry in range(Num_Entry):
    EventId = WaveformTable[entry]['EventID']
    ChannelId = WaveformTable[entry]['ChannelID']
    Waveform = WaveformTable[entry]['Waveform']
    Wave_Vec = make_wave_long_vec(Waveform)
    EventMat.append(EventId)
    ChanMat.append(ChannelId)
    WaveMat.append(Wave_Vec)

    # periodic save to avoid memory crash
    if (entry+1) % save_period ==0:
        EventData = np.array(EventMat)
        ChanData = np.array(ChanMat)
        WaveData = np.array(WaveMat)
        name = SavePath+'prediction_'+str(start_entry)+'-'+str(entry)
        np.savez(name, Event=EventData, Chan=ChanData, Wave=WaveData)
        start_entry = entry+1
        EventMat = []
        ChanMat = []
        WaveMat = []
        print("Data Saved")

    # check point
    if entry%5000==0:
        print(entry)

# Data Save for the final round
EventData = np.array(EventMat)
ChanData = np.array(ChanMat)
WaveData = np.array(WaveMat)
name = SavePath+'prediction_'+str(start_entry)+'-'+str(Num_Entry-1)
np.savez(name, Event=EventData, Chan=ChanData, Wave=WaveData)
print("Data Saved")
print("Prediction Pre-processing Finished")

h5file.close()

