#!/usr/bin/env python3

import sys
if len(sys.argv) != 2 :
    print("Usage: python3 runNo")
    exit(1)
else :
    runNo = int(sys.argv[1])

import os
CalibDir = os.getenv('JPCalibDir') + "/GainCalibData/"
import pandas as pd

mastertable = pd.read_csv(CalibDir + "Master.GainCalibMap.txt", sep=r'[ ]+', engine="python", skiprows=1, header=None)
larger = mastertable[0] <= runNo
smaller = mastertable[1] >= runNo
whichrow = smaller & larger
if(whichrow.any()) : print(CalibDir + mastertable[4][whichrow].to_list()[0] + ".txt")
elif((~smaller).all()) : print(CalibDir + mastertable[4][len(mastertable) - 1] + ".txt")
elif((~larger).all()) : print(CalibDir + mastertable[4][0] + ".txt")
else : raise ValueError("runNo={0} cannot be found in {1}Master.GainCalibMap.txt".format(runNo, CalibDir))
