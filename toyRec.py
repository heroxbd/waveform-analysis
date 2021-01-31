# -*- coding: utf-8 -*-

import time
import argparse
from functools import partial
from multiprocessing import Pool, cpu_count

import h5py
import numpy as np
import scipy.optimize as opti

import wf_func as wff

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', type=str, help='output file')
psr.add_argument('ipt', type=str, help='input file')
psr.add_argument('--Ncpu', dest='Ncpu', type=int, default=100)
psr.add_argument('--ref', type=str, help='reference file')
args = psr.parse_args()

global_start = time.time()
cpu_global_start = time.process_time()

window = 1029
npe = 10
gmu = 160.

def probcharhitt(t0, hitt, char):
    prob = np.sum([wff.probcharge(char, i, gmu=gmu) * np.power(wff.convolve_exp_norm(hitt - t0, Tau, Sigma), i) for i in range(1, npe)]) / np.sum([wff.probcharge(char, i, gmu=gmu) for i in range(1, npe)])
    return prob

probch = np.vectorize(probcharhitt, excluded=['t0'])

def start_time(a0, a1, mode):
    stime = np.empty(a1 - a0)
    for i in range(a0, a1):
        hitt = charge[i_cha[i]:i_cha[i+1]]['HitPosInWindow'].astype(np.float)
        char = charge[i_cha[i]:i_cha[i+1]]['Charge']
        b = [0., 600.]
        tlist = np.arange(b[0], b[1])
        if mode == 'charge':
            logL = lambda t0 : -1 * np.sum(np.log(np.clip(probch(t0, hitt, char), np.finfo(np.float).tiny, np.inf)))
        elif mode == 'all':
            logL = lambda t0 : -1 * np.sum(np.log(np.clip(wff.convolve_exp_norm(pelist[i_pel[i]:i_pel[i+1]]['HitPosInWindow'] - t0, Tau, Sigma), np.finfo(np.float).tiny, np.inf)))
        logLv = np.vectorize(logL)
        stime[i - a0] = opti.fmin_l_bfgs_b(logL, x0=[tlist[np.argmin(logLv(tlist))]], approx_grad=True, bounds=[b], maxfun=500000)[0]
    return stime

spe_pre = wff.read_model('spe.h5')
with h5py.File(args.ipt, 'r', libver='latest', swmr=True) as ipt, h5py.File(args.ref, 'r', libver='latest', swmr=True) as ref:
    pelist = ref['SimTriggerInfo/PEList'][:]
    charge = ipt['photoelectron'][:]
    Tau = ipt['photoelectron'].attrs['tau']
    Sigma = ipt['photoelectron'].attrs['sigma']
pelist = np.sort(pelist, kind='stable', order=['TriggerNo', 'PMTId', 'HitPosInWindow'])
Chnum = len(np.unique(charge['ChannelID']))
charge = np.sort(charge, kind='stable', order=['TriggerNo', 'ChannelID', 'HitPosInWindow'])
e_pel = pelist['TriggerNo'] * Chnum + pelist['PMTId']
e_pel, i_pel = np.unique(e_pel, return_index=True)
i_pel = np.append(i_pel, len(pelist))

e_cha = charge['TriggerNo'] * Chnum + charge['ChannelID']
e_cha, i_cha = np.unique(e_cha, return_index=True)
N = len(i_cha)
i_cha = np.append(i_cha, len(charge))
assert np.all(e_cha == e_pel), 'File not match!'

sdtp = np.dtype([('TriggerNo', np.uint32), ('ChannelID', np.uint32), ('tsfirsttruth', np.float64), ('tsfirstcharge', np.float64), ('tstruth', np.float64), ('tscharge', np.float64)])
ts = np.zeros(N, dtype=sdtp)
ts['TriggerNo'] = np.unique(pelist['TriggerNo'])
ts['ChannelID'] = np.unique(pelist['PMTId'])
ts['tsfirsttruth'] = np.array([np.min(pelist[i_pel[i]:i_pel[i+1]]['HitPosInWindow']) for i in range(N)])
ts['tsfirstcharge'] = np.array([np.min(charge[i_cha[i]:i_cha[i+1]]['HitPosInWindow']) for i in range(N)])

chunk = N // args.Ncpu + 1
slices = np.vstack((np.arange(0, N, chunk), np.append(np.arange(chunk, N, chunk), N))).T.astype(np.int).tolist()
with Pool(min(args.Ncpu, cpu_count())) as pool:
    result = pool.starmap(partial(start_time, mode='all'), slices)
ts['tstruth'] = np.hstack(result)
with Pool(min(args.Ncpu, cpu_count())) as pool:
    result = pool.starmap(partial(start_time, mode='charge'), slices)
ts['tscharge'] = np.hstack(result)

with h5py.File(args.opt, 'w') as opt:
    dset = opt.create_dataset('risetime', data=ts, compression='gzip')
    print('The output file path is {}'.format(args.opt))

print('Finished! Consuming {0:.2f}s in total, cpu time {1:.2f}s.'.format(time.time() - global_start, time.process_time() - cpu_global_start))