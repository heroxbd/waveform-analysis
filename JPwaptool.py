# -*- coding: utf-8 -*-

import numpy as np
from scipy.fftpack import fft, ifft

MAX_ADC = 1023

n4M3 = np.array([[-0.0117845, 0.00589226, 0.0109428, 0.00757576, 1.38826e-18, -0.00757576, -0.0109428, -0.00589226, 0.0117845], 
                 [0.030303, 0.00757576, -0.00865801, -0.0183983, -0.021645, -0.0183983, -0.00865801, 0.00757576,     0.030303], 
                 [0.0723906, -0.119529, -0.162458, -0.106061, -1.91768e-17, 0.106061, 0.162458, 0.119529, -0.0723906], 
                 [-0.0909091, 0.0606061, 0.168831, 0.233766, 0.255411, 0.233766, 0.168831, 0.0606061, -0.0909091]])

class ChannelInfo:
    def __init__(self):
        self.ChannelId = -1
        self.Pedestal = 0
        self.PedestalStd = 0
        self.BslnMean = 0
        self.BslnStd = 0
        self.Charge = 0
#         self.FullCharge = 0
#         self.Peak = 0
#         self.RiseTime = 0
#         self.PE = 0
        self.ChargeMaskLen = 0
        self.PedMaskLen = 0
#         self.nPeak_in_frnt = 0
#         self.HitTime = np.array([]).astype(np.int)
        self.PeakLoc = np.array([]).astype(np.int)
        self.PeakAmp = np.array([])
        return
    
class JPwaptool:
    def __init__(self, WindowSize, bl_end=50, inte_end=400, frnt_blur=10, back_blur=50):
        self.ChannelInfo = ChannelInfo
        self.WindowSize = WindowSize
        self.ped_upperlimit = 0
        self.ped_lowerlimmit = 0
        self.frnt_blur = frnt_blur
        self.back_blur = back_blur
        self.bl_end = bl_end
        self.inte_end = inte_end
        self.Baseline = 0
        self.ThreMask = np.empty(self.WindowSize, dtype=np.bool)
        self.ChargeMask = np.empty(self.WindowSize, dtype=np.bool)
        self.PedMask = np.empty(self.WindowSize, dtype=np.bool)
        return
    
    def Return_Result(self):
        return self.ChannelInfo.Pedestal, self.ChannelInfo.PedestalStd, self.ChannelInfo.Charge, self.ChannelInfo.PedMaskLen, self.ChannelInfo.ChargeMaskLen
    
    def GetThreMask(self, data):
        histogram = np.zeros(self.WindowSize+1, dtype=np.int)
        t, c = np.unique(data, return_counts=True)
        histogram[t] = c

        most_ADC = np.argmax(histogram)
        most_ADC_statistic = histogram[most_ADC]
        bottom = 1
        top = MAX_ADC
        while histogram[bottom]<most_ADC_statistic/2.51164:
            bottom += 1
        while histogram[top]<most_ADC_statistic/2.51164:
            top -= 1
        count = np.sum(histogram[np.arange(bottom, top+1)])
        ped_mean = np.sum(histogram[np.arange(bottom, top+1)]*np.arange(bottom, top+1))
        ped_std = np.sum(histogram[np.arange(bottom, top+1)]*np.arange(bottom, top+1)*np.arange(bottom, top+1))
        ped_mean = ped_mean / count
        ped_std = np.sqrt(ped_std/count - ped_mean*ped_mean)
        ped_std = max(ped_std*2.20308, 1.)
        self.ped_lowerlimit = int(np.around(ped_mean - ped_std))
        self.ped_upperlimit = int(np.around(ped_mean + ped_std))
        self.ThreMask = data < self.ped_lowerlimit
        return
    
    def ExpandMask(self, back_blur):
        self.ChargeMask = self.ThreMask.copy()
        i = 1
        while i < self.WindowSize:
            if self.ThreMask[i-1] and self.ThreMask[i]:
                self.ChargeMask[max(i-self.frnt_blur, 0):i-1] = True
                while i < self.WindowSize and self.ThreMask[i]:
                    i += 1
                self.ChargeMask[i:min(i+back_blur, self.WindowSize)] = True
            i += 1
        return
    
    def Dynamic_ExpandMask(self):
        max_chargemask_len = self.WindowSize - self.bl_end
        back_blur_start = self.back_blur
        back_blur_step = 5
        back_blur_stop = 20
        if back_blur_start < back_blur_stop:
            back_blur_stop = back_blur_start
        while back_blur_start >= back_blur_stop:
            back_blur_start -= back_blur_step
            self.ExpandMask(back_blur_start)
            if np.sum(self.ChargeMask) < max_chargemask_len: 
                break
        return back_blur_start
    
    def GetPedinfo(self, data):
        self.ChannelInfo.PedMaskLen = 0
        self.ChannelInfo.BslnMean = 0
        self.ChannelInfo.BslnStd = 0
        reach_bl_end = False
        for i in range(self.WindowSize):
            if self.ChargeMask[i]:
                self.PedMask[i] = False
                continue
            if data[i] > self.ped_upperlimit:
                self.PedMask[i] = False
                continue
            self.PedMask[i] = True
            self.ChannelInfo.BslnMean += data[i]
            self.ChannelInfo.BslnStd += data[i]*data[i]
            self.ChannelInfo.PedMaskLen += 1
            if reach_bl_end:
                continue
            if (i >= self.bl_end and self.ChannelInfo.PedMaskLen >= self.WindowSize/15) or i == self.WindowSize-1:
                self.ChannelInfo.Pedestal = self.ChannelInfo.BslnMean
                self.ChannelInfo.PedestalStd = self.ChannelInfo.BslnStd
                count = self.ChannelInfo.PedMaskLen
        if self.ChannelInfo.PedMaskLen == 0:
            self.ChannelInfo.BslnMean = np.mean(data)
            self.ChannelInfo.BslnStd = np.std(data)
        else:
            self.ChannelInfo.BslnMean = self.ChannelInfo.BslnMean / self.ChannelInfo.PedMaskLen
            self.ChannelInfo.BslnStd = self.ChannelInfo.BslnStd / self.ChannelInfo.PedMaskLen - self.ChannelInfo.BslnMean * self.ChannelInfo.BslnMean
        if count == 0:
            self.ChannelInfo.Pedestal = np.mean(data[:50])
            self.ChannelInfo.PedestalStd = np.std(data[:50])
        else:
            self.ChannelInfo.Pedestal = self.ChannelInfo.Pedestal / count
            self.ChannelInfo.PedestalStd = np.sqrt(self.ChannelInfo.PedestalStd / count - self.ChannelInfo.Pedestal * self.ChannelInfo.Pedestal)
        return
    
    def GetCharge(self, data, debug=False):
        self.ChannelInfo.ChargeMaskLen = 0
        charge = 0
        i = self.bl_end
        while self.ChargeMask[i]:
            if i == -1:
                break
            i -= 1
        i += 1
        while i < self.inte_end:
            if self.ChargeMask[i]:
                charge += data[i]
                self.ChannelInfo.ChargeMaskLen += 1
                if debug:
                    print('{}: {}, {}: {}'.format(i, data[i], self.ChannelInfo.ChargeMaskLen, charge))
            i += 1
        while self.ChargeMask[i]:
            if i == self.WindowSize:
                charge += data[i]
                self.ChannelInfo.ChargeMaskLen += 1
                if debug:
                    print('{}: {}, {}: {}'.format(i, data[i], self.ChannelInfo.ChargeMaskLen, charge))
            i += 1
        if debug:
            print(self.ChannelInfo.Pedestal*self.ChannelInfo.ChargeMaskLen-charge)
        return self.ChannelInfo.Pedestal*self.ChannelInfo.ChargeMaskLen-charge
    
    def FindPeaksSG(self, data):
        z = fft(data)
        length = (self.WindowSize+1)//2
        z[(length-int(length*0.8)):(length+int(length*0.8))] = 0
        data2 = ifft(z).real
        n = 4
        M = 3
        peakList = np.array([]).astype(np.int)
        diff = np.zeros(self.WindowSize)
        diff2 = np.zeros(self.WindowSize)
        for i in range(self.WindowSize):
            for k in range(-n, n+1):
                aa = 0
                if i+k<0 or i+k>self.WindowSize-1:
                    aa = self.ChannelInfo.Pedestal
                else:
                    aa = data2[i+k]
                diff[i] += aa*n4M3[M-1][k+n]
                diff2[i] += aa*n4M3[M-2][k+n]
        for i in range(self.WindowSize-1):
            wv = data[i]
            b1 = False
            b2 = False
            if diff[i]<0 and diff[i+1]>0:
                b1 = True
            if i>0 and (diff[i-1]<0 and abs(diff[i])<1e-6 and diff[i+1]>0):
                b2 = True
            if b1 or b2:
                sd = diff2[max(i-5, 0):min(i+6, self.WindowSize)][np.argmax(diff2[max(i-5, 0):min(i+6, self.WindowSize)])]
                if sd < 0.1:
                    continue
                peakAmp1 = data2[max(i-2, 0):min(i+3, self.WindowSize)][np.argmin(data2[max(i-2, 0):min(i+3, self.WindowSize)])]
                peakLoc = np.arange(max(i-2, 0), min(i+3, self.WindowSize))[np.argmin(data[max(i-2, 0):min(i+3, self.WindowSize)])]
                peakAmp2 = data[peakLoc]
                if self.ChannelInfo.Pedestal - peakAmp1 > 3*self.ChannelInfo.PedestalStd and self.ChannelInfo.Pedestal - peakAmp2 > 3*self.ChannelInfo.PedestalStd:
                    flat_l = False
                    flat_r = False
                    if i < self.WindowSize - 5:
                        flat_r = True
                        for j in range(5):
                            vv = data[i+j]
                            if np.abs(vv-wv) > 1e-6:
                                flat_r = False
                                break
                    if i > 5:
                        flat_l = True
                        for j in range(5):
                            vv = data[i-j]
                            if np.abs(vv-wv) > 1e-6:
                                flat_l = False
                                break
                    if flat_l and flat_r:
                        continue
                    i += 3
                    if len(peakList) == 0 or peakList[-1] != peakLoc:
                        peakList = np.append(peakList, peakLoc)
        if len(peakList) == 0:
            mindataLoc = np.argmin(data2)
            if self.ChannelInfo.Pedestal - data2[mindataLoc] > 3*self.ChannelInfo.PedestalStd:
                minIter = np.argmin(data[max(mindataLoc-10, 0):min(mindataLoc+10, self.WindowSize)])
                peakList = np.append(peakList, minIter)
        return peakList
    
    def Calculate(self, data):
        data = data.astype(np.int32)
        self.GetThreMask(data)
        self.Dynamic_ExpandMask()
        self.GetPedinfo(data)
        self.ChannelInfo.Charge = self.GetCharge(data)
        peakList = self.FindPeaksSG(data)
        self.ChannelInfo.PeakLoc = np.array([]).astype(np.int)
        self.ChannelInfo.PeakAmp = np.array([])
        for i in peakList:
            peakamp = self.ChannelInfo.Pedestal - data[i]
            if peakamp <= 0:
                continue
            self.ChannelInfo.PeakLoc = np.append(self.ChannelInfo.PeakLoc, i)
            self.ChannelInfo.PeakAmp = np.append(self.ChannelInfo.PeakAmp, peakamp)
        if len(self.ChannelInfo.PeakLoc) == 0:
            item = np.argmin(data)
            self.ChannelInfo.PeakLoc = np.append(self.ChannelInfo.PeakLoc, item)
            self.ChannelInfo.PeakAmp = np.append(self.ChannelInfo.PeakAmp, self.ChannelInfo.Pedestal - data[item])
        return
    
    def FastCalculate(self, data):
        data = data.astype(np.int32)
        self.GetThreMask(data)
        self.Dynamic_ExpandMask()
        self.GetPedinfo(data)
        self.ChannelInfo.Charge = self.GetCharge(data)
        return
