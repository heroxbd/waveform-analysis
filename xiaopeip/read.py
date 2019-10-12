import h5py
import tables
import numpy as np
import sys
args=int(sys.argv[1])

# Read hdf5 file
h5file = tables.open_file(f"./data/atraining-{args}.h5", "r")
WaveformTable = h5file.root.Waveform
GroundTruthTable = h5file.root.GroundTruth

sinevet,sinchan,sintime=[],[],[]

#根据groundtruth找出只有单光子的事例
i=1
while i <100000:
    if GroundTruthTable[i]['ChannelID']!=GroundTruthTable[i-1]['ChannelID'] and GroundTruthTable[i]['ChannelID']!=GroundTruthTable[i+1]['ChannelID']:
        sinevet.append(GroundTruthTable[i]['EventID'])
        sintime.append(GroundTruthTable[i]['PETime'])
        sinchan.append(GroundTruthTable[i]['ChannelID'])
    i+=1

#将单光子事例波形累加
sumwave=np.zeros(1029,dtype=np.int32)
sinlen=len(sinevet)
for x in range(sinlen):
    if x%100==0:
        print(f"{x*100/sinlen}%")
    posi=0
    while True:
        if WaveformTable[posi]["EventID"]==sinevet[x] and WaveformTable[posi]["ChannelID"]==sinchan[x]:
            break
        posi+=1
    sumwave+=np.append(WaveformTable[posi]['Waveform'][sintime[x]:],WaveformTable[posi]['Waveform'][:sintime[x]])-972

#求得平均值
averwave=sumwave/sinlen
averzero=np.average(averwave[100:])
spe=averwave-averzero

with h5py.File(f"medium/average{args+1}.h5", "w") as opt1:
    opt1.create_dataset("averzero", data=np.array([averzero]))
with h5py.File(f'medium/singlewave{args+1}.h5',"w") as opt2:
    opt2.create_dataset("spe",data=spe,compression="gzip", shuffle=True)
#写入文件

h5file.close()
