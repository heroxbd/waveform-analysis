import tables
import h5py
import numpy as np

class AnswerData(tables.IsDescription):
    EventID    = tables.Int64Col(pos=0)
    ChannelID  = tables.Int16Col(pos=1)
    PETime     = tables.Int16Col(pos=2)
    Weight     = tables.Float32Col(pos=3)

# Create the output file and the group
h5file = tables.open_file("result/Total.h5", mode="w", title="OneTonDetector")

# Create tables
AnswerTable = h5file.create_table("/", "Answer", AnswerData, "Answer")
answer = AnswerTable.row

for i in range(1,1798):
    print(i)
    with h5py.File("result/"+f"{i}"+"-pgan.h5") as ipt:
        hg=ipt["Answer"][()]
# Write data 
    for j in range(len(hg)):
        answer['EventID'] =  hg[j,0]
        answer['ChannelID'] = hg[j,1]
        answer['PETime'] = hg[j,2]
        answer['Weight'] = hg[j,3]
        answer.append()

# Flush into the output file
AnswerTable.flush()

h5file.close()