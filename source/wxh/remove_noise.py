#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

# define function to devide waveform list into groups


def divide(ls, unit_len):
    DividedLs = []
    UnitLen = int(unit_len)
    GroupCount = int(len(ls) / UnitLen)
    GroupCountExact = len(ls) / UnitLen
    Start = 0
    for i in range(GroupCount):
        DividedLs.append(ls[Start: (Start + UnitLen)])
        Start = Start + UnitLen
    if GroupCount < GroupCountExact:  # put all remainders into the last group
        DividedLs.append(ls[GroupCount * UnitLen:])
    return DividedLs

# define function to get width of division


def width(lst):
    i = 0
    for j in lst[0]:
        i += 1
    return i

# define function to get average


def GetAverage(mat):
    average = []
    for group in mat:
        average.append(np.mean(group))
    return average


# define function to get standard deviation
def GetStd(mat):
    std = []
    for group in mat:
        std.append(np.std(group, ddof=1))
    return std

# define function to remove noise in waveform


def DenoisMat(mat, StdJudge):
    average = GetAverage(mat)
    std = GetStd(mat)
    n = len(mat)
    m = width(mat)
    # replace small fluctuations with average
    for i in range(n-1):
        if std[i] < StdJudge:
            mat[i] = [average[i]] * m
    if std[-1] < StdJudge:
        mat[-1] = [average[-1]] * len(mat[-1])
    m = width(mat)
    num = [0] * m
    # merge into one list
    DenoisMat = []
    for i in range(n):
        for j in mat[i]:
            DenoisMat.append(j)
    return DenoisMat


def dwave(i):
    wave = i['Waveform']
    std = np.std(wave, ddof=1)
    StdJudge = std / 2
    unit_len = 3
    mat = divide(wave, unit_len)
    wave = DenoisMat(mat, StdJudge)
    return wave


if __name__ == '__main__':
    # runtime
    pool = mp.Pool(32)
    
    # input file
    with h5py.File(sys.argv[1], 'r') as ipt:
        wf = ipt['Waveform']
        # replace small unit fluctuations with average
        DenoisWave = pool.map(dwave, tqdm(wf))
        # output data
        denois = np.zeros(len(wf), dtype=wf.dtype)
        denois['EventID'] = wf['EventID']
        denois['ChannelID'] = wf['ChannelID']
        denois['Waveform'] = DenoisWave
    
    # output file
    with h5py.File(sys.argv[2], 'w') as opt:
        opt.create_dataset('Waveform', data=denois)
