# -*- coding: utf-8 -*-
"""
Created on Thu May  9 11:39:38 2019

@author: liveFreeOrDie
"""

import pandas as pd
import tables
import numpy as np

answer_ld = pd.read_csv("MyAnswer.txt") ;  # please modify the input .txt as the name specified by the matlab...
answer_ld = answer_ld.values

len(answer_ld)
answer_ld.shape

class AnswerData(tables.IsDescription):
    EventID = tables.Int64Col(pos=0)
    ChannelID = tables.Int16Col(pos=1)
    PETime = tables.Int16Col(pos=2)
    Weight = tables.Float32Col(pos=3)
    

h5file = tables.open_file("myAnswer.h5", mode = "w", title = "OneTonDetector")

AnswerTable = h5file.create_table("/","Answer", AnswerData, "Answer")
answer = AnswerTable.row
for x in range(0, answer_ld.shape[0]-1): 
    answer['EventID'] = answer_ld[x,0]
    answer['ChannelID'] = answer_ld[x,1]
    answer['PETime'] = answer_ld[x,2]
    answer['Weight'] = answer_ld[x,3]
    answer.append()

AnswerTable.flush()
h5file.close()



