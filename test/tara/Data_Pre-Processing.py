import time
time_start = time.time()
import numpy as np
import os
import sys
import tables

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
    max_set_number = min(max_set_number),len(Len_Entry))
else :
    max_set_number = Len_Entry

print(Len_Entry, "data entries") # Entry 10^6

# Make Data Matrix
def make_time_long_vec(time_mark_vec,WindowSize):
    time_long_vec = np.zeros(WindowSize)
    time_long_vec = np.zeros(WindowSize)
    for time_mark in time_mark_vec:
        if time_mark < WindowSize-9:
            time_long_vec[time_mark] += 1
    if len(time_mark_vec) == 0:
        print("non-PET-event")
    return np.array(time_long_vec,dtype=np.float32)

def make_wave_long_vec(wave_form):
    # non_negative_peak + zero_base_level
    shift = np.argmax(np.bincount(wave_form)) #make baseline shifts, normally 972
    shift = np.mean(wave_form[np.abs(wave_form-shift)<3])
    shift_wave=np.array(wave_form-shift,dtype=np.int16)
    #non_negative_peak + zero_base_level
    if np.max(shift_wave) >= -np.min(shift_wave) : return shift_wave
    if np.max(shift_wave) < -np.min(shift_wave) : return -shift_wave

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

WindowSize = len(WaveformTable[0]['Waveform'])

class PreProcessedData(tables.IsDescription):
    PET = tables.Col.from_type('int16', shape=WindowSize, pos=0)
    Wave = tables.Col.from_type('int16', shape=WindowSize, pos=1)
    
#create Pre-Processed output file
Prefile = tables.open_file(SavePath+"Pre.h5", mode="w", title="Pre-Processed-Training-Data")

# Create group and tables
group = "/"
TrainDataTable = Prefile.create_table(group, "TrainDataTable", PreProcessedData, "Wave and PET")
traindata = TrainDataTable.row

for entry in range(Num_Entry):
    EventId = WaveformTable[entry]['EventID']
    ChannelId = WaveformTable[entry]['ChannelID']
    Waveform = WaveformTable[entry]['Waveform']
    PETime, looking_up_index = search_PETime(EventId,ChannelId,GroundTruthTable,looking_up_index)
    traindata['PET'] = make_time_long_vec(np.array(PETime,dtype=np.int16),WindowSize)
    traindata['Wave'] = make_wave_long_vec(Waveform)
    traindata.append()

    # periodic save to avoid memory crash
    if (entry+1) % save_period ==0 or entry==Num_Entry:
        TrainDataTable.flush()
        print("Data Saved")

    # check point
    if (entry+1)%5000==0:
        print(entry+1)

h5file.close()
Prefile.close()
time_end = time.time()
print('consuming time: {}s'.format(time_end-time_start))
