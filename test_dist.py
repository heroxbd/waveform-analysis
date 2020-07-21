# -*- coding: utf-8 -*-

import sys
import numpy as np
import csv
import h5py
import scipy.stats
import itertools as it
import argparse
from multiprocessing import Pool, cpu_count
import wf_func as wff

psr = argparse.ArgumentParser()
psr.add_argument('--ref', dest='ref', help='reference file', nargs='+')
psr.add_argument('ipt', help="input file")
psr.add_argument('--mod', type=str, help='mode of pe or charge')
psr.add_argument('-N', dest='Ncpu', type=int, help='cpu number', default=50)
psr.add_argument('-o', dest='opt', help='output file')
psr.add_argument('-p', dest='print', action='store_false', help='print bool', default=True)
args = psr.parse_args()

if args.print:
    sys.stdout = None

fref = args.ref[0]
fipt = args.ipt
fopt = args.opt
mode = args.mod
Ncpu = args.Ncpu

def wpdist(a, b):
    for i, c in zip(range(a, b), range(b-a)):
        cid = df_wav[i_wav[i]]['ChannelID']
        wave = wff.deduct_base(spe_pre[cid]['epulse'] * df_wav[i_wav[i]]['Waveform'], spe_pre[cid]['m_l'], spe_pre[cid]['thres'], 20, 'detail')
        
        wl = df_sub[i_sub[i]:i_sub[i+1]][mode]
        pet_sub = df_sub[i_sub[i]:i_sub[i+1]]['PETime']
        pf_s = np.zeros(leng); pf_s[pet_sub] = wl
        wave1 = np.convolve(spe_pre[cid]['spe'], pf_s, 'full')[:leng]
        if mode == 'Weight':
            pet0, pwe0 = np.unique(df_ans[i_ans[i]:i_ans[i+1]]['PETime'], return_counts=True)
            pf0 = np.zeros(leng); pf0[pet0] = pwe0
            wave0 = np.convolve(spe_pre[cid]['spe'], pf0, 'full')[:leng]
            Q = i_ans[i+1]-i_ans[i]; q = np.sum(wl)
            dt['pdist'][c] = np.abs(Q - q) * scipy.stats.poisson.pmf(Q, Q)
            dt['PEnum'][c] = Q
        elif mode == 'Charge':
            pet0 = df_ans[i_ans[i]:i_ans[i+1]]['RiseTime']; pwe0 = df_ans[i_ans[i]:i_ans[i+1]][mode] 
            pf0 = np.zeros(leng); pf0[pet0] = pwe0
            wave0 = np.convolve(spe_pre[cid]['spe'], pf0, 'full')[:leng] / np.sum(spe_pre[cid]['spe'])
            wave1 = wave1 / np.sum(spe_pre[cid]['spe'])
            dt['PEnum'][c] = len(pet0)

        dt['wdist'][c] = scipy.stats.wasserstein_distance(pet0, pet_sub, u_weights=pwe0, v_weights=wl)
        dt['EventID'][c] = df_wav[i_wav[i]]['EventID']//Chnum
        dt['ChannelID'][c] = cid
        dt['RSS_truth'][c] = np.power(wave0 - wave, 2).sum()
        dt['RSS_recon'][c] = np.power(wave1 - wave, 2).sum()
        if 'PEdiff' in df_sub.dtype.names:
            dt['PEdiff'][c] = df_sub[i_sub[i]]['PEdiff']
    return dt

spe_pre = wff.read_model(args.ref[1])
with h5py.File(fref, 'r', libver='latest', swmr=True) as ref, h5py.File(fipt, 'r', libver='latest', swmr=True) as ipt:
    df_ans = ref['GroundTruth'][:]
    df_wav = ref['Waveform'][:]
    df_sub = ipt['Answer'][:]
    method = ipt['Answer'].attrs['Method']
df_ans = np.sort(df_ans, kind='stable', order=['EventID', 'ChannelID'])
df_sub = np.sort(df_sub, kind='stable', order=['EventID', 'ChannelID'])
df_wav = np.sort(df_wav, kind='stable', order=['EventID', 'ChannelID'])
Chnum = len(np.unique(df_ans['ChannelID']))
e_ans = df_ans['EventID']*Chnum + df_ans['ChannelID']
e_ans, i_ans = np.unique(e_ans, return_index=True)
i_ans = np.append(i_ans, len(df_ans))

gl = len(e_ans); leng = len(df_wav[0]['Waveform'])
opdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint32), ('PEnum', np.uint16), ('wdist', np.float32), ('pdist', np.float32), ('RSS_recon', np.float32), ('RSS_truth', np.float32), ('PEdiff', np.float32)])
dt = np.zeros(gl, dtype=opdt); dt['pdist'] = np.nan; dt['PEdiff'] = np.nan

e_wav = df_wav['EventID']*Chnum + df_wav['ChannelID']; df_wav = df_wav[np.isin(e_wav, e_ans)]
e_wav, i_wav = np.unique(df_wav['EventID']*Chnum + df_wav['ChannelID'], return_index=True)

e_sub = df_sub['EventID']*Chnum + df_sub['ChannelID']; df_sub = df_sub[np.isin(e_sub, e_ans)]
e_sub, i_sub = np.unique(df_sub['EventID']*Chnum + df_sub['ChannelID'], return_index=True)
i_sub = np.append(i_sub, len(df_sub))
assert len(e_ans) ==  len(e_wav) and len(e_ans) == len(e_sub), 'Incomplete Submission'

l = len(e_sub); chunk = l // Ncpu + 1
slices = np.vstack((np.arange(0, l, chunk), np.append(np.arange(chunk, l, chunk), l))).T.astype(np.int).tolist()
with Pool(min(Ncpu, cpu_count())) as pool:
    result = pool.starmap(wpdist, slices)
dt = np.hstack(result)
with h5py.File(fopt, 'w') as h5f:
    dset = h5f.create_dataset('Record', data=dt, compression='gzip')
    dset.attrs['Method'] = method
