import time
time_start = time.time()
import tables
import numpy as np
import os
import sys

## Initialization
fullfilename = sys.argv[1]
SavePath = sys.argv[2]
# Make Directory
if not os.path.exists(SavePath):
    os.makedirs(SavePath)
# Read hdf5 file

h5file = tables.open_file(fullfilename, "r")
WaveformTable = h5file.root.Waveform
Len_Entry = len(WaveformTable)
print(Len_Entry, "data entries") # Entry 10^6
# WaveChannel 0-1028, length 600

# Make Data Matrix
def make_wave_long_vec(wave_form):
    # non_negative_peak + zero_base_level
    shift = np.argmax(np.bincount(wave_form)) #make baseline shifts, normally 972
    shift = np.mean(wave_form[np.abs(wave_form-shift)<3])
    shift_wave=np.array(wave_form-shift,dtype=np.int16)
    #non_negative_peak + zero_base_level
    if np.max(shift_wave) >= -np.min(shift_wave) : return shift_wave
    if np.max(shift_wave) < -np.min(shift_wave) : return -shift_wave

# Data Pre-Processing
Num_Entry = Len_Entry #100000 #Len_Entry
EventMat=[]
ChanMat=[]
WaveMat=[]
save_period = 200000

WindowSize = len(WaveformTable[0]['Waveform'])

class PreProcessedData(tables.IsDescription):
    EventID    = tables.Int64Col(pos=0)
    ChannelID  = tables.Int16Col(pos=1)
    Waveform   = tables.Col.from_type('int16', shape=WindowSize, pos=2)
#create Pre-Processed output file
Prefile = tables.open_file(SavePath+"Pre.h5", mode="w", title="Pre-Processed-Training-Data")

# Create group and tables
group = "/"
TestDataTable = Prefile.create_table(group, "TestDataTable", PreProcessedData, "Wave and PET")
testdata = TestDataTable.row
#saving params
start_entry = 0
entry_index = 0


for entry in range(Num_Entry):
    testdata["EventID"] = WaveformTable[entry]['EventID']
    testdata["ChannelID"] = WaveformTable[entry]['ChannelID']
    Waveform = WaveformTable[entry]['Waveform']
    testdata["Waveform"] = make_wave_long_vec(Waveform)
    testdata.append()

    # periodic save to avoid memory crash
    if (entry+1) % save_period ==0 or entry==Num_Entry:
        TestDataTable.flush()
        print("Data Saved")

    # check point
    if entry%5000==0:
        print(entry)
print("Prediction Pre-processing Finished")

h5file.close()
Prefile.close()
time_end = time.time()
print('consuming time: {}s'.format(time_end-time_start))
