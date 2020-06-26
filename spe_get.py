# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import argparse
import itertools as it
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
h5_path = args.ipt
single_pe_path = args.opt
if args.print:
    sys.stdout = None

def mean(dt):
    Chnum = np.unique(dt['ChannelID'])
    N = 10
    cid = np.zeros(len(Chnum))
    spemean = np.zeros((len(Chnum), len(dt[0]['speWf'])-N))
    for i in range(len(Chnum)):
        dt_cid = dt[dt['ChannelID']==Chnum[i]]
        spemean_i = np.mean(dt_cid['speWf'], axis=0)
        base_vol = np.mean(spemean_i[-N:])
        spemean_i = spemean_i[:-N] - base_vol
        if np.sum(spemean_i) > 0:
            epulse = 1
        else:
            epulse = -1
            spemean_i = epulse * spemean_i
        spemean_i = np.where(spemean_i > 0.001, spemean_i, 0)
        spemean[i] = spemean_i
    return spemean, epulse, Chnum

def pre_analysis(spemean, epulse, Wf):
    Chnum = np.unique(Wf['ChannelID'])
    thres = np.zeros(len(Chnum))
    for i in range(len(Chnum)):
        Wf_cid = Wf[Wf['ChannelID']==Chnum[i]]
        sam = wff.deduct_base(epulse * Wf_cid['Waveform'], mode='fast').flatten()
        a_std = np.std(np.sort(sam)[:-sam.shape[0]//3])
        t = 0
        r = 3
        j = 0
        while np.abs(t - a_std) > 0.01:
            a = 0
            b = 0
            t = a_std
            dt = np.zeros(N).astype(np.float128)
            while True:
                wave = wff.deduct_base(Wf_cid[j]['Waveform'], mode='fast')
                j = (j + 1)%len(Wf_cid)
                vali = wff.vali_base(wave, np.sum(spemean[i] > r*a_std), r*a_std)
                a = b
                b = a + np.sum(vali == 0)
                if b >= N:
                    dt[a:] = wave[vali==0][:N-a]-np.mean(wave[vali==0])
                    break
                else:
                    dt[a:b] = wave[vali==0]-np.mean(wave[vali==0])
            a_std = np.std(dt, ddof=1).astype(np.float64)
        thres[i] = r*a_std
    spe_pre = {'spe':spemean, 'thres':thres}
    return spe_pre

def generate_standard(h5_path, single_pe_path):
    npdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint32), ('speWf', np.uint16, L)])  # set datatype
    dt = np.zeros(N, dtype=npdt)
    num = 0

    with h5py.File(h5_path[0], 'r', libver='latest', swmr=True) as ztrfile:
        Gt = ztrfile['GroundTruth']
        Wf = ztrfile['Waveform']
        Chnum = len(np.unique(Gt['ChannelID']))
        e_gt = Gt['EventID']*Chnum + Gt['ChannelID']
        e_gt, i_gt = np.unique(e_gt, return_index=True)
        e_wf = Wf['EventID']*Chnum + Wf['ChannelID']
        e_wf, i_wf = np.unique(e_wf, return_index=True)
        leng = len(Wf[0]['Waveform'])
        p = 0
        for e_gt_i, a, b in zip(e_gt, np.nditer(i_gt), it.chain(np.nditer(i_gt[1:]), [len(Gt)])):
            while e_wf[p] < e_gt_i:
                p = p + 1
            pt = np.sort(Gt[a:b]['PETime']).astype(np.int)
            pt = pt[pt >= 0]
            if pt.shape[0] != 0:
                if len(pt) == 1 and pt[0] < leng - L:
                    ps = pt
                else:
                    dpta = np.diff(pt, prepend=pt[0])
                    dptb = np.diff(pt, append=pt[-1])
                    ps = pt[np.logical_and(dpta > L, dptb > L)]#long distance to other spe in both forepart & backpart
                if ps.shape[0] != 0:
                    for k in range(len(ps)):
                        dt[num]['EventID'] = Wf[i_wf[p]]['EventID']
                        dt[num]['ChannelID'] = Wf[i_wf[p]]['ChannelID']
                        dt[num]['speWf'] = Wf[i_wf[p]]['Waveform'][ps[k]:ps[k]+L]
                        num += 1
                        if num >= N:
                            break
            print('\rSingle PE Generating:|{}>{}|{:6.2f}%'.format(((20*(num-1))//N)*'-', (19 - (20*(num-1))//N)*' ', 100 * (num / N)), end=''if num != N else '\n')
            if num >= N or b == len(Gt):
                dt = dt[:num] # cut empty dt part
                dt = np.sort(dt, kind='stable', order=['EventID', 'ChannelID'])
                if Chnum < 100:
                    assert Chnum == len(np.unique(dt['ChannelID']))
                else:
                    dt['ChannelID'] = 0
                print('{} speWf generated'.format(len(dt)))
                spemean, epulse, cid = mean(dt)
                spe_pre = pre_analysis(spemean, epulse, Wf[:10000])
                break
    with h5py.File(single_pe_path, 'w') as spp:
        dset = spp.create_dataset('SinglePE', data=dt)
        dset.attrs['SpePositive'] = spe_pre['spe']
        dset.attrs['Epulse'] = epulse
        dset.attrs['Thres'] = spe_pre['thres']
        dset.attrs['ChannelID'] = cid

if not os.path.exists(single_pe_path):
    generate_standard(h5_path, single_pe_path) # generate response model
