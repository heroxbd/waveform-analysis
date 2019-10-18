import h5py
import matplotlib.pyplot as plt
import numpy as np
import math
import sys

# fipt = "zincm-problem.h5"
# fopt = "submissionfile.h5"
fipt = sys.argv[1]
fopt = sys.argv[2]

opd = [('EventID', '<i8'), ('ChannelID', '<i2'), ('PETime', 'f4'), ('Weight', 'f4')]
tresh = 9 #设定阈值，超过该阈值的记作一个PE信号
para = [4.70557863824900, 8.59153064595592, 39.5970278962013, 4.17111271556104] #预定义的函数参数

#拟合一个响应函数
def modelfunc(a1,a2,a3,a4,t):
    if (t <= a2 and t >= a1):
        return a3 * (1 - math.exp(-(t - a1) / a4))
    elif (t > a2):
        return a3 * (math.exp(-(a1 - a2) / a4) - 1) * math.exp(-(t - a1) / a4)
    else:
        return 0
#将该函数矢量化
vecmodelfunc = np.vectorize(modelfunc)

def findpeak(w): #寻找波形的极大值点并返回最大值点的位置，如有平坦的极大值区则寻找极大值区下降沿
    w1 = np.sign(np.diff(w))
    I1 = np.array(range(0,len(w1)))
    I2 = I1[w1!=0]
    w2 = w1[I2]
    w2 = np.diff(w2)
    w2 = np.insert(w2,0,0)
    Ip = I2[w2==-2]
    if len(Ip)==0:
        return []
    else:
        return Ip[0]

def mmp(wr):
    w = np.array(wr[2], dtype=np.int16) #波形

    nothing = np.array(w[range(0,200)], dtype=np.int16) #选取前200ns波形算基准
    backg = np.average(nothing) 
    nothing = nothing[nothing-backg < 6]
    backg = np.average(nothing) 
    wr2 = np.round(backg) - w #减去基准

    tot = 0
    finalpetime = []
    weigh = []

    if (wr2 < tresh).all(): #如果没有任何波形过阈值，则选取波形最高点作为PEtime，同时权重为1
        tot = np.argmax(wr2) #找到波形最高点时间为tot
        finalpetime.append(tot - 7) #设定偏移量为7
        weigh.append(1) #权重为1
    else:
        begining = wr2[range(0,10)] < tresh #检测前10ns波形有没有过阈（没有则全为1）
        if not begining.all(): #并非全为1，即有
            tot = begining[begining == False] #如果前10ns中有过阈信号
            finalpetime = [tot[0] - 6] #找到它，设置偏移
            weigh.append(1)
            wr2[range(1,10)] = 0 #然后把前10ns信号设置为0。上述操作是防止有半个波形卡在前10ns导致程序bug。

        while True: #循环：减去每个找到的波形
            wr3 = wr2 #拷贝波形用于处理
            wr3[wr3 < tresh] = 0 #未过阈点认为是噪声，设为0
            tot = np.argwhere(wr3 > 0) #找到过阈时间tot
            if len(tot) == 0: #如果再也找不到，宣告已全部找出所有信号
                break
        
            #使用第一个tot
            petime = tot[0][0] - 6 #如果找到了，那么减去偏移量得到这个信号的petime
            
            peaktime = findpeak(wr3)
            if len(peaktime) == 0: #如果找不到极大值，则说明信号已经延申到波形外，认为是最后一个波形
                weigh.append(1) #权重都为1
                finalpetime.append(petime) #写入一个光电子时间及其权重
                break

            #一些cut条件
            if len(finalpetime) == 0 or (petime - finalpetime[-1]) < 3 or (petime - finalpetime[-1]) > 5 or wr2[peaktime] > 12:
                weigh.append(1) #权重都为1
                finalpetime = list(finalpetime)
                # print(type(finalpetime))
                
                finalpetime.append(petime) #写入一个光电子时间及其权重
            if len(finalpetime) > 500: #如果发现了超过有500个petime，应该是陷入了死循环，立即中止并报错。
                raise Exception('too many PEs found, there must be a bug.')

            dis = np.arange(1029)
            wr2 -= np.round(vecmodelfunc(petime + para[0], petime + para[1], para[2], para[3], dis))
    
    rst = np.zeros(len(finalpetime), dtype=opd)
    rst['PETime'] = np.array(finalpetime)+1
    rst['Weight'] = np.array(weigh)
    rst['EventID'] = wr[0]
    rst['ChannelID'] = wr[1]

    return rst

#测试用代码
#ipt = h5py.File(fipt)
#wr = ipt['Waveform'][100]
#d = mmp(wr)
#plt.plot(wr['Waveform'])
#plt.vlines(d['PETime'],900,1000)
#plt.show()

#
with h5py.File(fipt) as ipt,h5py.File(fopt, "w") as opt:
    dt = np.concatenate([mmp(wr) for wr in ipt['Waveform'][:]]) #先完整读入文件再写会快一点
    opt.create_dataset('Answer', data=dt, compression='gzip')
