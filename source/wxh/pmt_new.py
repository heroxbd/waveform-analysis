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
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import tables
import multiprocessing as mp


with h5py.File(sys.argv[1],'r') as ipt:
    wf = ipt['Waveform'][...]
w = pd.DataFrame({'EventID':wf['EventID'],'ChannelID':wf['ChannelID']})
w['Waveform'] = list(wf['Waveform'])
# 对event及channel进行遍历
event = int(sys.argv[2])
channel = int(sys.argv[3])


for i in w.to_records():
    if i['EventID'] == event and i['ChannelID'] == channel:
        wave = i['Waveform']
ped = np.mean(wave[:150])
# 反转波形
wave = ped - wave
petime = []
weight = []
# 阈值选取
threshold = 3
wp = signal.find_peaks_cwt(wave,np.arange(1,5))
for i in wp:
    if wave[i] > threshold:
        weight.append(wave[i]/threshold)
        petime.append(i-wave[i]/threshold)

dt = [('EventID','int64'),('ChannelID','int16'),('PETime','int16'),('Weight','float64')]
ans = np.zeros(len(petime),dtype = dt)
ans['EventID'] = np.array([event]*len(petime))
ans['ChannelID'] = np.array([channel]*len(petime))
ans['PETime'] = petime
ans['Weight'] = weight
# 输出文件：每个event及channel暂时存入一个文件
with h5py.File('output_{}{}.h5'.format(event,channel),'w') as opt:
        opt.create_dataset('GroundTruth',data = ans)

