#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 输入文件数据
# 减去基线，反转波形
# 寻找峰位及确定权重
# 转化为起始位
# 输出数据

import sys
import pandas as pd
import numpy as np
import h5py
import scipy
from scipy import signal
# import matplotlib as mpl
# mpl.use('Agg')
# from matplotlib import pyplot as plt
import multiprocessing as mp
from tqdm import tqdm

# define function to find peaks
def findpeak(info):
    event = info['EventID']
    channel = info['ChannelID']
    wave = info['Waveform']
    ped = np.mean(wave[:150])
    # 反转波形
    wave = ped - wave
    # 阈值选取
    threshold = 3
<<<<<<< HEAD
    wp = signal.find_peaks_cwt(wave,np.arange(1,5))

    eve = []
    chan = []
    petime = []
    weight = []
    for i in wp:
        if wave[i] > threshold:
            weight.append(wave[i] / threshold)
            petime.append(i - wave[i] / threshold)
    eve.append([event] * len(petime))
    chan.append([channel] * len(petime))
    return eve, chan, petime, weight
=======
    wp = signal.find_peaks(wave)
    event = []
    channel = []
    for i in wp[0]:
        if wave[i] > threshold:
            weight.append(wave[i] / threshold)
            petime.append(i - wave[i] / threshold)
    event.append([event] * len(petime))
    channel.append([channel] * len(petime))
    return event, channel
>>>>>>> refs/remotes/origin/master

if __name__ == '__main__':
    # input file
    with h5py.File(sys.argv[1],'r') as ipt:
        wf = ipt['Waveform'][...]
    w = pd.DataFrame({'EventID':wf['EventID'],'ChannelID':wf['ChannelID']})
    w['Waveform'] = list(wf['Waveform'])
<<<<<<< HEAD
    ans_e = []
    ans_c = []
    ans_p = []
    ans_w = []
    pool = mp.Pool(32)
    ans_f = pool.map(findpeak, tqdm(w.to_records(index = False)))
    
    for e,c,p,w in ans_f:
        ans_e.extend(e)
        ans_c.extend(c)
        ans_p.extend(p)
        ans_w.extend(w)
=======
    petime = []
    weight = []

    pool = mp.Pool(32)
    output = pool.map(findpeak, tqdm(w.to_records(index=False)))
    ans_e, ans_c = zip(*output)

>>>>>>> refs/remotes/origin/master
    # save into numpy array
    dt = [('EventID','int64'), ('ChannelID','int16'), ('PETime','int16'), ('Weight','float64')]
    ans = np.zeros(len(ans_c), dtype = dt)
    ans['EventID'] = ans_e
    ans['ChannelID'] = ans_c
    ans['PETime'] = ans_p
    ans['Weight'] = ans_w

    # 输出文件
    with h5py.File(sys.argv[2],'w') as opt:
        opt.create_dataset('GroundTruth',data = ans)
    ans['PETime'] = petime
    ans['Weight'] = weight

    # 输出文件
    with h5py.File(sys.argv[2],'w') as opt:
        opt.create_dataset('GroundTruth',data = ans)

