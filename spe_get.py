# -*- coding: utf-8 -*-

import os
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import argparse
import itertools as it
from JPwaptool import JPwaptool
import wf_func as wff

psr = argparse.ArgumentParser()
psr.add_argument('ipt', nargs='+', help='input file')
psr.add_argument('-o', dest='opt', help='output file')
psr.add_argument('--num', dest='spenum', type=int, help='num of speWf', default=5e5)
psr.add_argument('--len', dest='spelen', type=int, help='length of speWf', default=80)
args = psr.parse_args()

N = args.spenum
L = args.spelen
h5_path = args.ipt
single_pe_path = args.opt

def mean(dt):
    Chnum = np.unique(dt['ChannelID'])
    cid = np.zeros(len(Chnum))
    spemean = np.zeros((len(Chnum), L))
    for i in range(len(Chnum)):
        dt_cid = dt[dt['ChannelID']==Chnum[i]]
        spemean_i = np.mean(dt_cid['speWf'], axis=0)
        if np.median(spemean_i) > 0:
            epulse = 1
        else:
            epulse = -1
            spemean_i = epulse * spemean_i
        spemean_i = np.where(spemean_i > 0.001, spemean_i, 0)
        spemean[i] = spemean_i
    return spemean, epulse, Chnum

def pre_analysis(spemean, stddt):
    Chnum = np.unique(stddt['ChannelID'])
    thres = np.zeros(len(Chnum))
    for i in range(len(Chnum)):
        stddt_cid = stddt[stddt['ChannelID']==Chnum[i]]['PedWave']
        thres[i] = 3 * np.std(stddt_cid, ddof=-1)
    spe_pre = {'spe':spemean, 'thres':thres}
    return spe_pre

def generate_standard(h5_path, single_pe_path):
    npdt = np.dtype([('ChannelID', np.uint32), ('speWf', np.float64, L)])  # set datatype
    dt = np.zeros(N, dtype=npdt)
    num = 0

    with h5py.File(h5_path[0], 'r', libver='latest', swmr=True) as ztrfile:
        Gt = ztrfile['GroundTruth'][:]
        Wf = ztrfile['Waveform'][:]
    Gt = np.sort(Gt, kind='stable', order=['EventID', 'ChannelID'])
    Wf = np.sort(Wf, kind='stable', order=['EventID', 'ChannelID'])
    Chnum = len(np.unique(Gt['ChannelID']))
    e_gt, i_gt = np.unique(Gt['EventID']*Chnum + Gt['ChannelID'], return_index=True)
    i_gt = np.append(i_gt, len(Gt))
    e_wf, i_wf = np.unique(Wf['EventID']*Chnum + Wf['ChannelID'], return_index=True)
    Wf = Wf[np.isin(e_wf, e_gt)]
    e_wf, i_wf = np.unique(Wf['EventID']*Chnum + Wf['ChannelID'], return_index=True)
    assert len(e_wf) ==  len(e_gt), 'Incomplete Dataset'
    leng = len(Wf[0]['Waveform'])
    p = 0
    stream = JPwaptool(leng, 150, 600, 7, 15)
    for p in tqdm(range(len(e_wf))):
        pt = np.sort(Gt[i_gt[p]:i_gt[p+1]]['RiseTime']).astype(np.int)
        if len(pt) == 1:
            ps = pt
        else:
            dpta = np.diff(pt, prepend=pt[0])
            dptb = np.diff(pt, append=pt[-1])
            ps = pt[(dpta > L) & (dptb > L)]#long distance to other spe in both forepart & backpart
        ps = ps[(ps >= 0) & (ps < leng - L)]
        if ps.shape[0] != 0:
            stream.Calculate(Wf[i_wf[p]]['Waveform'])
            wave = Wf[i_wf[p]]['Waveform'] - stream.ChannelInfo.Pedestal
            for k in range(len(ps)):
                dt[num]['ChannelID'] = Wf[i_wf[p]]['ChannelID']
                dt[num]['speWf'] = wave[ps[k]:ps[k]+L]
                num += 1
                if num >= N:
                    break
        if num >= N or p == len(e_wf)-1:
            dt = dt[:num] # cut empty dt part
            if Chnum < 100:
                assert Chnum == len(np.unique(dt['ChannelID']))
            else:
                dt['ChannelID'] = 0
            print('{} speWf generated'.format(len(dt)))
            break
    
    npstddt = np.dtype([('ChannelID', np.uint32), ('PedWave', np.float64)])  # set datatype
    panel = np.arange(0, leng)
    stddt = np.zeros(N * 10, dtype=npstddt); stddt['PedWave'] = np.nan
    start = 0
    for p in tqdm(range(len(e_wf))):
        pt = np.sort(Gt[i_gt[p]:i_gt[p+1]]['RiseTime']).astype(np.int)
        c = np.concatenate(([np.arange(i, i+L) for i in pt]))
        c = np.unique(np.clip(c, 0, leng))
        c = panel[np.logical_not(np.isin(panel, c))]
        stream.Calculate(Wf[i_wf[p]]['Waveform'])
        wave = Wf[i_wf[p]]['Waveform'] - stream.ChannelInfo.Pedestal
        end = start + len(c)
        stddt[start:min(end, N * 10)]['ChannelID'] = Wf[i_wf[p]]['ChannelID']
        stddt[start:min(end, N * 10)]['PedWave'] = wave[c][:min(len(c), N * 10 - start)]
        start = end
        if end >= N * 10:
            break
    stddt = stddt[np.logical_not(np.isnan(stddt['PedWave']))]
    spemean, epulse, cid = mean(dt)
    assert len(cid) == len(np.unique(stddt['ChannelID'])), 'Incomplete PedWave'
    spe_pre = pre_analysis(spemean, stddt)
    with h5py.File(single_pe_path, 'w') as spp:
        dset = spp.create_dataset('SinglePE', data=dt)
        dset.attrs['SpePositive'] = spe_pre['spe']
        dset.attrs['Epulse'] = epulse
        dset.attrs['Thres'] = spe_pre['thres']
        dset.attrs['ChannelID'] = cid

if not os.path.exists(single_pe_path):
    generate_standard(h5_path, single_pe_path) # generate response model
