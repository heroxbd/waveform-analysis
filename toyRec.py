# -*- coding: utf-8 -*-

import time
import argparse
from functools import partial
from multiprocessing import Pool, cpu_count

import h5py
import numpy as np
from numpy.lib import recfunctions
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
npe = 4
gmu = 160.

def probcharhitt(t0, hitt, probcharge):
    prob = probcharge * np.power(wff.convolve_exp_norm(hitt - t0, Tau, Sigma), np.arange(1, npe)[:, None])
    prob = np.sum(prob / np.sum(probcharge, axis=0), axis=0)
    return prob

def start_time(a0, a1, mode):
    stime = np.empty(a1 - a0)
    for i in range(a0, a1):
        b = [0., 600.]
        tlist = np.arange(b[0], b[1] + 1)
        if mode == 'charge':
            hitt = charge[i_cha[i]:i_cha[i+1]]['HitPosInWindow'].astype(np.float)
            char = charge[i_cha[i]:i_cha[i+1]]['Charge']
            probcharge = np.array([wff.probcharge(char, i, gmu=gmu) for i in range(1, npe)])
            logL = lambda t0 : -1 * np.sum(np.log(np.clip(probcharhitt(t0, hitt, probcharge), np.finfo(np.float).tiny, np.inf)))
            # logL = lambda t0 : -1 * np.sum(np.log(np.clip(wff.convolve_exp_norm(hitt - t0, Tau, Sigma) * (char / gmu), np.finfo(np.float).tiny, np.inf)))
        elif mode == 'all':
            hitt = pelist[i_pel[i]:i_pel[i+1]]['HitPosInWindow'].astype(np.float)
            logL = lambda t0 : -1 * np.sum(np.log(np.clip(wff.convolve_exp_norm(hitt - t0, Tau, Sigma), np.finfo(np.float).tiny, np.inf)))
        logLv = np.vectorize(logL)
        stime[i - a0] = opti.fmin_l_bfgs_b(logL, x0=[tlist[np.argmin(logLv(tlist))]], approx_grad=True, bounds=[b], maxfun=500000)[0]
    return stime

spe_pre = wff.read_model('spe.h5')
with h5py.File(args.ipt, 'r', libver='latest', swmr=True) as ipt, h5py.File(args.ref, 'r', libver='latest', swmr=True) as ref:
    pelist = ref['SimTriggerInfo/PEList'][:]
    method = ipt['photoelectron'].attrs['Method']
    Mu = ipt['photoelectron'].attrs['mu']
    Tau = ipt['photoelectron'].attrs['tau']
    Sigma = ipt['photoelectron'].attrs['sigma']
    charge = ipt['photoelectron'][:]
    Chnum = len(np.unique(charge['ChannelID']))
    charge = np.sort(charge, kind='stable', order=['TriggerNo', 'ChannelID', 'HitPosInWindow'])
    e_cha = charge['TriggerNo'] * Chnum + charge['ChannelID']
    e_cha, i_cha = np.unique(e_cha, return_index=True)
    i_cha = np.append(i_cha, len(charge))
    N = len(e_cha)
    pelist = np.sort(pelist, kind='stable', order=['TriggerNo', 'PMTId', 'HitPosInWindow'])
    e_pel = pelist['TriggerNo'] * Chnum + pelist['PMTId']
    e_pel, i_pel = np.unique(e_pel, return_index=True)
    i_pel = np.append(i_pel, len(pelist))
    tcdtp = np.dtype([('TriggerNo', np.uint32), ('ChannelID', np.uint32)])
    tc = np.zeros(len(pelist), dtype=tcdtp)
    tc['TriggerNo'] = pelist['TriggerNo']
    tc['ChannelID'] = pelist['PMTId']
    tc = np.unique(tc)
    iftswave = 'starttime' in ipt
    if iftswave:
        tswave = ipt['starttime'][:]
    else:
        sdtp = np.dtype([('TriggerNo', np.uint32), ('ChannelID', np.uint32), ('tswave', np.float64)])
        tswave = np.zeros(N, dtype=sdtp)
        tswave['TriggerNo'] = tc['TriggerNo']
        tswave['ChannelID'] = tc['ChannelID']
        tswave['tswave'] = np.full(N, np.nan)
    assert np.all(e_cha == e_pel), 'File not match!'

sdtp = np.dtype([('TriggerNo', np.uint32), ('ChannelID', np.uint32), ('tsfirsttruth', np.float64), ('tstruth', np.float64), ('tscharge', np.float64)])
ts = np.zeros(N, dtype=sdtp)
ts['TriggerNo'] = tc['TriggerNo']
ts['ChannelID'] = tc['ChannelID']
ts['tsfirsttruth'] = np.array([np.min(pelist[i_pel[i]:i_pel[i+1]]['HitPosInWindow']) for i in range(N)])

chunk = N // args.Ncpu + 1
slices = np.vstack((np.arange(0, N, chunk), np.append(np.arange(chunk, N, chunk), N))).T.astype(np.int).tolist()
with Pool(min(args.Ncpu, cpu_count())) as pool:
    result = pool.starmap(partial(start_time, mode='all'), slices)
ts['tstruth'] = np.hstack(result)
print('Mode all finished, real time {0:.4f}s, cpu time {1:.4f}s until now'.format(time.time() - global_start, time.process_time() - cpu_global_start))
with Pool(min(args.Ncpu, cpu_count())) as pool:
    result = pool.starmap(partial(start_time, mode='charge'), slices)
ts['tscharge'] = np.hstack(result)
ts = recfunctions.join_by(('TriggerNo', 'ChannelID'), ts, tswave, usemask=False)
print('Mode charge finished, real time {0:.4f}s, cpu time {1:.4f}s until now'.format(time.time() - global_start, time.process_time() - cpu_global_start))

with h5py.File(args.opt, 'w') as opt:
    dset = opt.create_dataset('starttime', data=ts, compression='gzip')
    dset.attrs['Method'] = method
    dset.attrs['mu'] = Mu
    dset.attrs['tau'] = Tau
    dset.attrs['sigma'] = Sigma
    print('The output file path is {}'.format(args.opt))

print('Finished! Consuming {0:.02f}s in total, cpu time {1:.02f}s.'.format(time.time() - global_start, time.process_time() - cpu_global_start))