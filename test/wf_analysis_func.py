# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

def estipulse(h5_path):
    with h5py.File(h5_path, 'r', libver='latest', swmr=True) as ztrfile:
        Wf = ztrfile['Waveform']['Waveform']
        flag = 0
        i = 0
        while flag == 0:
            mm = np.ptp(Wf[i])
            mean = np.mean(Wf[i])
            if np.max(Wf[i]) - mean > 3/4*mm:
                flag = 1
            if mean - np.min(Wf[i]) > 3/4*mm:
                flag = -1
            i += 1
    return flag

def generate_model(spe_path, epulse):
    with h5py.File(spe_path, 'r', libver='latest', swmr=True) as speFile:
        spemean = np.mean(speFile['SinglePE']['speWf'], axis=0)
        base_vol = np.mean(spemean[-10:])
        # stdmodel[0] is the single pe's incoming time
        stdmodel = spemean[:-10] - base_vol
        # stdmodel = np.around(stdmodel / 0.05) * 0.05 # smooth the stdmodel
        # cut off all small values
        if epulse == -1:
            stdmodel = np.where(stdmodel < -0.02, stdmodel, 0)
        elif epulse == 1:
            stdmodel = np.where(stdmodel > 0.02, stdmodel, 0)
        # stdmodel = np.where(stdmodel >= 0, stdmodel, 0) # cut off all negative values
    return stdmodel

def speplot(dt):
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 16  # set figure parameters

    # calculate average fluctuation of waveform
    spemean = np.mean(dt['speWf'], axis=0)
    plt.figure()
    plt.xlim(0, 50)
    plt.plot(spemean)  # draw the average fluctuation
    plt.xlabel('ns')
    plt.ylabel('mV')
    plt.title("Standard response model")
    # plt.savefig('spemean.png')
    plt.close()

    spemin = np.min(dt['speWf'], axis=1)
    u = np.unique(Pt)
    plt.figure()
    # show the dispersion of minimum of spe waveform
    plt.hist(spemin, len(u), density=1, histtype='bar', cumulative=False)
    plt.xlabel('mV')
    # plt.savefig('specumu.png')
    plt.close()
    return

def snip_baseline(waveform, itera=10):
    wm = np.min(waveform)
    waveform = waveform - wm
    v = np.log(np.log(np.sqrt(waveform+1)+1)+1)
    N = len(waveform)
    for i in range(itera):
        v[i:N-i] = np.minimum(v[i:N-i], (v[:N-2*i] + v[2*i:])/2)
    w = np.power(np.exp(np.exp(v) - 1) - 1, 2) - 1 + wm
    return w

def pre_analysis(h5_path, epulse, spemean):
    peak_c = np.argmin(spemean)
    zero_l = np.where(spemean[peak_c:] == 0)[0][0] + peak_c
    N = 10000
    t = 0
    with h5py.File(h5_path, 'r', libver='latest', swmr=True) as ztrfile:
        Wf = ztrfile['Waveform']['Waveform']
        n = 2*N//len(Wf[0])+1
        sam = Wf[:n].flatten()
        a_std = np.std(np.sort(sam)[len(sam)//10:])
        r = 3
        i = 0
        while np.abs(t - a_std) > 0.01:
            a = 0
            b = 0
            t = a_std
            dt = np.zeros(N).astype(np.float128)
            while True:
                wave = -1*epulse*Wf[i]
                wave = deduct_base(wave, mode='fast')
                i = (i + 1)%len(Wf)
                vali = vali_base(wave, np.sum(spemean < -r*a_std), -r*a_std)
                a = b
                b = a + np.sum(vali == 0)
                if b >= N:
                    dt[a:] = wave[vali==0][:N-a]-np.mean(wave[vali==0])
                    break
                else:
                    dt[a:b] = wave[vali==0]-np.mean(wave[vali==0])
            a_std = np.std(dt, ddof=1)
    thres = -r*a_std
    m_l = np.sum(spemean < thres)
    mar_l = np.sum(spemean[:peak_c] > thres) + 2
    mar_r = np.sum(spemean[peak_c:zero_l] > thres) + 2
    return peak_c, zero_l, m_l, mar_l, mar_r, thres

def vali_base(waveform, m_l, thres):
    m = np.median(waveform[waveform > np.median(waveform)])
    vali = np.where(m - waveform > -1*thres, 1, 0) # valid waveform, not dark noise
    pos = omi2pos(vali)
    pos = rm_frag(pos, m_l)
    vali = pos2omi(pos, len(waveform))
    return vali

def deduct_base(waveform, m_l=None, thres=None, itera=10, mode='fast'):
    wf_flip = -1*waveform
    baseline = snip_baseline(wf_flip, itera)
    wave = baseline - wf_flip
    if mode == 'detail':
        wave = wave - find_base(wave, m_l, thres)
    elif mode == 'fast':
        wave = wave - find_base_fast(wave)
    return wave

def find_base(waveform, m_l, thres):
    vali = vali_base(waveform, m_l, thres)
    base_line = np.mean(waveform[vali == 0])
    return base_line

def find_base_fast(waveform):
    m = np.median(waveform[waveform > np.median(waveform)])
    base_line = np.mean(waveform[np.logical_and(waveform < m + 4, waveform > m - 4)])
    return base_line

def omi2pos(vali):
    vali_t = np.concatenate((np.array([0]), vali, np.array([0])), axis=0)
    dval = np.diff(vali_t)
    pos_begin = np.argwhere(dval == 1).flatten()
    pos_end = np.argwhere(dval == -1).flatten()
    pos = np.concatenate((pos_begin.reshape(len(pos_begin), 1), pos_end.reshape(len(pos_end), 1)), axis = 1).astype(np.int16)
    return pos

def pos2omi(pos, len_n):
    vali = np.zeros(len_n).astype(np.int16)
    for i in range(len(pos)):
        vali[pos[i][0]:pos[i][1]] = 1
    return vali

def rm_frag(pos, m_l):
    n = len(pos)
    pos_t = []
    for i in range(n):
        if pos[i][1] - pos[i][0] > m_l:
            pos_t.append(pos[i])
    pos = np.array(pos_t)
    return pos
