# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import argparse
import wf_func as wff

psr = argparse.ArgumentParser()
psr.add_argument('ipt', nargs='+', help='input file')
psr.add_argument('-o', dest='opt', help='output file')
psr.add_argument('--num', dest='spenum', type=int, help='num of speWf')
psr.add_argument('--len', dest='spelen', type=int, help='length of speWf')
psr.add_argument('-p', dest='print', action='store_false', help='print bool', default=True)
args = psr.parse_args()

N = args.spenum
L = args.spelen
if args.print:
    sys.stdout = None

def mean(dt):
    spemean = np.mean(dt['speWf'], axis=0)
    base_vol = np.mean(spemean[-10:])
    # stdmodel[0] is the single pe's incoming time
    stdmodel = spemean[:-10] - base_vol
    if np.sum(stdmodel) > 0:
        epulse = 1
        stdmodel = -1 * stdmodel
    else:
        epulse = -1
    # cut off all small values
    stdmodel = np.where(stdmodel < -0.01, stdmodel, 0)
    peak_i = np.argmin(stdmodel)
    a = 0
    b = stdmodel.shape[0]
    for _ in range(b):
        if not np.all(stdmodel[a:peak_i] < 0):
            a = a + 1
        if not np.all(stdmodel[peak_i:b] < 0):
            b = b - 1
    spemean = np.zeros_like(stdmodel[:b])
    spemean[a:b] = stdmodel[a:b]
    spemean = -1 * epulse * spemean
    return spemean, epulse

def pre_analysis(spemean, Wf):
    sam = Wf.flatten()
    a_std = np.std(np.sort(sam)[:-sam.shape[0]//3])
    t = 0
    r = 3
    i = 0
    while np.abs(t - a_std) > 0.01:
        a = 0
        b = 0
        t = a_std
        dt = np.zeros(N).astype(np.float128)
        while True:
            wave = wff.deduct_base(Wf[i], mode='fast')
            i = (i + 1)%Wf.shape[0]
            vali = wff.vali_base(wave, np.sum(spemean > r*a_std), r*a_std)
            a = b
            b = a + np.sum(vali == 0)
            if b >= N:
                dt[a:] = wave[vali==0][:N-a]-np.mean(wave[vali==0])
                break
            else:
                dt[a:b] = wave[vali==0]-np.mean(wave[vali==0])
        a_std = np.std(dt, ddof=1).astype(np.float64)
    thres = r*a_std
    spe_pre = {'spe':spemean, 'thres':thres}
    return spe_pre

def generate_standard(h5_path, single_pe_path):
    npdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint32), ('speWf', np.uint16, L)])  # set datatype
    dt = np.zeros(N, dtype=npdt)
    num = 0

    with h5py.File(h5_path[0], 'r', libver='latest', swmr=True) as ztrfile:
        ptev = ztrfile['GroundTruth']['EventID']
        ptch = ztrfile['GroundTruth']['ChannelID']
        Pt = ztrfile['GroundTruth']['PETime']
        wfev = ztrfile['Waveform']['EventID']
        wfch = ztrfile['Waveform']['ChannelID']
        Wf = ztrfile['Waveform']['Waveform']
        Length_pe = len(Wf[0])
        for j in range(len(Wf)):
            wf = Wf[j]
            pt = np.sort(Pt[np.logical_and(ptev == wfev[j], ptch == wfch[j])]).astype(np.int)
            pt = pt[pt >= 0]
            if len(pt) == 1 and pt[0] < Length_pe - L:
                ps = pt
            else:
                dpta = np.diff(pt, prepend=pt[0])
                dptb = np.diff(pt, append=pt[-1])
                ps = pt[np.logical_and(dpta > L, dptb > L)]#long distance to other spe in both forepart & backpart
            for k in range(len(ps)):
                dt[num]['EventID'] = wfev[j]
                dt[num]['ChannelID'] = wfch[j]
                dt[num]['speWf'] = wf[ps[k]:ps[k]+L]
                print('\rSingle PE Generating:|{}>{}|{:6.2f}%'.format(((20*num)//N)*'-', (19 - (20*num)//N)*' ', 100 * ((num+1) / N)), end=''if num != N-1 else '\n')
                num += 1
                if num >= N:
                    break
            if num >= N:
                break
    dt = dt[:num] # cut empty dt part
    print('{} speWf generated'.format(len(dt)))
    spemean, epulse = mean(dt)
    spe_pre = pre_analysis(epulse * spemean, epulse * Wf[:1000])
    with h5py.File(single_pe_path, 'w') as spp:
        dset = spp.create_dataset('SinglePE', data=dt)
        dset.attrs['SpePositive'] = spe_pre['spe']
        dset.attrs['Epulse'] = epulse
        dset.attrs['Thres'] = spe_pre['thres']

def main(h5_path, single_pe_path):
    if not os.path.exists(single_pe_path):
        generate_standard(h5_path, single_pe_path) # generate response model

if __name__ == '__main__':
    main(args.ipt, args.opt)
