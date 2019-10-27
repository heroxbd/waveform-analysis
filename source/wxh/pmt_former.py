# 读取文件（hdf5）
# 去除噪声
# 寻找峰值和零点，得到PETime
# 将最小的峰值处设为权重1，分别得到不同点的权重
import sys
import pandas as pd
import numpy as np
import h5py
import scipy
from scipy import signal
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt


# 读取文件和建立dataframe
with h5py.File(sys.argv[1],'r') as ipt:
    wf = ipt['Waveform'][...]
w = pd.DataFrame({'EventID':wf['EventID'],'ChannelID':wf['ChannelID']})
w['Waveform'] = list(wf['Waveform'])
event = int(sys.argv[2])
channel = int(sys.argv[3])
# wave = w.query('EventID' == event and 'ChannelID' == channel")
for i in w.to_records():
    if i['EventID'] == event and i['ChannelID'] == channel:
        wave = i['Waveform']
wp = signal.find_peaks_cwt(-wave,np.arange(1,5))
st = {}
for i in wp:
    st[i] = wave[i]
mi = np.array(list(st.values())).max()
weight = {}
for i in wp:
    weight[i] = abs((970 - wave[i])/(970 - mi))

petime = []
pe_w = []
for i in wp:
    m = i-weight[i]
    if m != i:
        petime.append(m)
        pe_w.append(weight[i])
# pet = list(set(petime))
# pet.sort(key = petime.index)
# 去掉0
l = len(petime)
for i in range(l):
        if pe_w[i] < 1:
                del pe_w[i]
                del petime[i]
dt = [('EventID','int64'),('ChannelID','int16'),('PETime','int16'),('Weight','float64')]
ans_f = np.zeros(len(petime),dtype = dt)
ans_f['EventID'] = np.array([event]*len(petime))
ans_f['ChannelID'] = np.array([channel]*len(petime))
ans_f['PETime'] = petime
ans_f['Weight'] = pe_w



# 模拟，找到权重为1的情况
# 用峰值来确定权重，进行一个比较粗的模拟
# 讨论之后进行优化。


# 设置基线，计算权重，模拟与现实数据对比
# 找到击中时间
# 问题：训练数据
# plt.plot(w['Waveform'][0])
# plt.savefig('wave.png')


 
# # 寻找：单光子击中时间


# 频谱分析