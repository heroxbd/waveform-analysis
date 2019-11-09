import tables
import numpy as np
import os
import sys

## Initialization

# Make Directory
SavePath = "./Pre-Processing_Results_{}/".format(sys.argv[2])
if not os.path.exists(SavePath):
    os.makedirs(SavePath)
print("training data pre-processing savepath is {}".format(SavePath))

# Read hdf5 file
'''
LoadPath= "./DataSets/"
FileName = "ztraining-9"
'''
'''
LoadPath = '/home/greatofdream/killxbq/'
FileName = 'ftraining-0'
fullfilename = LoadPath+FileName+".h5"
'''
fullfilename = sys.argv[1]
FileName = os.path.basename(fullfilename).replace(".h5",'')
h5file = tables.open_file(fullfilename, "r")
WaveformTable = h5file.root.Waveform
GroundTruthTable = h5file.root.GroundTruth
Len_Entry = len(WaveformTable)
print(Len_Entry, "data entries") # Entry 10^6
# WaveChannel 0-1028, length 1029

# Make Data Matrix
def make_time_long_vec(time_mark_vec):
    time_long_vec = np.zeros(1029)
    for time_mark in time_mark_vec:
        if time_mark < 1020:
            time_long_vec[time_mark] += 1
    if len(time_mark_vec) == 0:
        print("non-PET-event")
    return time_long_vec

def make_wave_long_vec(wave_form):
    # non_negative_peak + zero_base_level
    shift= np.argmax(np.bincount(wave_form)) #make baseline shifts, normally 972
    shift_wave=-wave_form+shift #non_negative_peak + zero_base_level
    return shift_wave

def search_PETime(event_id,channel_id,PETtable,starting_point):
    PET_num = 0
    index = starting_point
    LEN = len(PETtable)
    PET_total = []
    while index < LEN and PETtable[index]['EventID'] == event_id and PETtable[index]['ChannelID'] == channel_id:
        PET_total.append(PETtable[index]['PETime'])
        PET_num += 1
        if index+1 < LEN:
            index += 1
        else:
            break
    ending_point = index
    return PET_total, ending_point


# Data Pre-Processing
Num_Entry = Len_Entry #100000 #Len_Entry


save_period = 200000
looking_up_index = 0

#saving params
start_entry = 0

EventMat=[]
ChanMat=[]
WaveMat=[]
PETMat=[]

for entry in range(Num_Entry):
    EventId = WaveformTable[entry]['EventID']
    ChannelId = WaveformTable[entry]['ChannelID']
    Waveform = WaveformTable[entry]['Waveform']
    PETime, looking_up_index = search_PETime(EventId,ChannelId,GroundTruthTable,looking_up_index)
    PETime_Vec = make_time_long_vec(PETime)
    Wave_Vec = make_wave_long_vec(Waveform)

    PETMat.append(PETime_Vec)
    WaveMat.append(Wave_Vec)

    # periodic save to avoid memory crash
    if (entry+1) % save_period ==0:
        WaveData = np.array(WaveMat)
        PETData = np.array(PETMat)
        name = SavePath+FileName+"_"+str(start_entry)+'-'+str(entry)
        np.savez(name, Wave=WaveData, PET=PETData)
        start_entry = entry+1
        WaveMat = []
        PETMat = []
        print("Data Saved")

    # check point
    if (entry+1)%5000==0:
        print(entry+1)


# Data Save for the final round
WaveData = np.array(WaveMat)
PETData = np.array(PETMat)
name = SavePath+FileName+"_"+str(start_entry)+'-'+str(Num_Entry)
np.savez(name, Wave=WaveData, PET=PETData)
print("Data Saved")
print("Pre-processing Finished")

h5file.close()

