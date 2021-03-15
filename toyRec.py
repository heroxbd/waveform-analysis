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
psr.add_argument('--ref', type=str, nargs='+', help='reference file')
args = psr.parse_args()

global_start = time.time()
cpu_global_start = time.process_time()

def start_time(a0, a1):
    stime = np.empty(a1 - a0)
    for i in range(a0, a1):
        hitt = charge[i_cha[i]:i_cha[i+1]]['HitPosInWindow'].astype(np.float64)
        char = charge[i_cha[i]:i_cha[i+1]]['Charge']
        t0, _ = wff.likelihoodt0(hitt, char=char, gmu=gmu, gsigma=gsigma, Tau=Tau, Sigma=Sigma, npe=npe, s0=s0, mode='charge')
        stime[i - a0] = t0
    return stime

spe_pre = wff.read_model(args.ref[1], wff.nshannon)
with h5py.File(args.ipt, 'r', libver='latest', swmr=True) as ipt, h5py.File(args.ref[0], 'r', libver='latest', swmr=True) as ref:
    npe = ref['SimTruth/T'].attrs['npe']
    gmu = ref['SimTriggerInfo/PEList'].attrs['gmu']
    gsigma = ref['SimTriggerInfo/PEList'].attrs['gsigma']
    s0 = spe_pre[0]['std'] / np.linalg.norm(spe_pre[0]['spe'])
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
    tcdtp = np.dtype([('TriggerNo', np.uint32), ('ChannelID', np.uint32)])
    tc = np.zeros(len(charge), dtype=tcdtp)
    tc['TriggerNo'] = charge['TriggerNo']
    tc['ChannelID'] = charge['ChannelID']
    tc = np.unique(tc)
    if 'starttime' in ipt:
        ts = ipt['starttime'][:]
    else:
        sdtp = np.dtype([('TriggerNo', np.uint32), ('ChannelID', np.uint32), ('tscharge', np.float64), ('tswave', np.float64)])
        ts = np.zeros(N, dtype=sdtp)
        ts['TriggerNo'] = tc['TriggerNo']
        ts['ChannelID'] = tc['ChannelID']
        ts['tswave'] = np.full(N, np.nan)

        chunk = N // args.Ncpu + 1
        slices = np.vstack((np.arange(0, N, chunk), np.append(np.arange(chunk, N, chunk), N))).T.astype(int).tolist()
        start_time(0, 10)
        with Pool(min(args.Ncpu, cpu_count())) as pool:
            result = pool.starmap(partial(start_time), slices)
        ts['tscharge'] = np.hstack(result)

with h5py.File(args.opt, 'w') as opt:
    dset = opt.create_dataset('starttime', data=ts, compression='gzip')
    dset.attrs['Method'] = method
    dset.attrs['mu'] = Mu
    dset.attrs['tau'] = Tau
    dset.attrs['sigma'] = Sigma
    dset.attrs['npe'] = npe
    print('The output file path is {}'.format(args.opt))

print('Finished! Consuming {0:.02f}s in total, cpu time {1:.02f}s.'.format(time.time() - global_start, time.process_time() - cpu_global_start))