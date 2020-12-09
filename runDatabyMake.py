#!/usr/bin/env python3

import sys
if len(sys.argv)<2 :
    print("Usage: python3 runData.py startRunNo endRunNo")
    print("Or:    python3 runData.py RunList.txt")
    exit(1)
import os
import numpy as np

JPDataDir = os.getenv('JPDataDir')
PreoutputDir = "../new_PreAnalysis"
os.system("mkdir -p "+PreoutputDir+"/log")

GoodRunList = set(np.loadtxt(os.environ["JPDataDir"]+"/GoodRunList.txt",skiprows=1,dtype=np.int))
if len(sys.argv)==2 :
    runrange = set(np.loadtxt(sys.argv[1],delimiter=", ",dtype=np.int))
else :
    runrange = set(np.arange(int(sys.argv[1]),int(sys.argv[2]),dtype=np.int))
runlist = np.array(list(GoodRunList & runrange))
runlist.sort()
print("{} jobs:".format(len(runlist)))
print(runlist)

with open("TargetList.txt","w") as f:
    for i in runlist :
        f.write("{:08d}\n".format(i))
