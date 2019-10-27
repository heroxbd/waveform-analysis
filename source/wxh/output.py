import h5py
import pandas as pd
import sys
with h5py.File(sys.argv[1],'r') as ipt:
    gd = ipt['GroundTruth'][...]
gdt = pd.DataFrame({'EventID':gd['EventID'],'ChannelID':gd['ChannelID'],'PETime':gd['PETime'],'Weight':gd['Weight']})

import tables

# Define the database columns
class AnswerData(tables.IsDescription):
    EventID    = tables.Int64Col(pos=0)
    ChannelID  = tables.Int16Col(pos=1)
    PETime     = tables.Int16Col(pos=2)
    Weight     = tables.Float32Col(pos=3)

# Create the output file and the group
h5file = tables.open_file("MyAnswer.h5", mode="w", title="OneTonDetector")

# Create tables
AnswerTable = h5file.create_table("/", "Answer", AnswerData, "Answer")
answer = AnswerTable.row

# Write data 
for i in gdt.to_records(index = False):
    i = list(i)
    answer['EventID'] = i[0]
    answer['ChannelID'] = i[1]
    answer['PETime'] = i[2]
    answer['Weight'] = i[3]
    answer.append()

# Flush into the output file
AnswerTable.flush()

h5file.close() 