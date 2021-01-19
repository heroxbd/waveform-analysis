# -*- coding: utf-8 -*-

import sys
import re
import time
import math
import argparse
import pickle
from functools import partial
from multiprocessing import Pool, cpu_count

import h5py
import numpy as np
# np.seterr(all='raise')
import pandas as pd
from tqdm import tqdm
import torch
import torch.distributions.constraints as constraints
import pyro
import pyro.distributions as dist
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc import NUTS
import matplotlib.pyplot as plt

import wf_func as wff

pyro.set_rng_seed(0)

global_start = time.time()
cpu_global_start = time.process_time()

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', type=str, help='output file')
psr.add_argument('ipt', type=str, help='input file')
psr.add_argument('--met', type=str, help='fitting method')
psr.add_argument('--ref', type=str, nargs='+', help='reference file')
psr.add_argument('-N', '--Ncpu', dest='Ncpu', type=int, default=50)
args = psr.parse_args()

fipt = args.ipt
fopt = args.opt
reference = args.ref
method = args.met

spe_pre = wff.read_model(reference[0])
with h5py.File(fipt, 'r', libver='latest', swmr=True) as ipt:
    ent = ipt['Readout/Waveform'][:]
    N = len(ent)
    print('{} waveforms will be computed'.format(N))
    window = len(ent[0]['Waveform'])
    assert window >= len(spe_pre[0]['spe']), 'Single PE too long which is {}'.format(len(spe_pre[0]['spe']))
    Mu = ipt['Readout/Waveform'].attrs['mu']
    Tau = ipt['Readout/Waveform'].attrs['tau']
    Sigma = ipt['Readout/Waveform'].attrs['sigma']
    pelist = ipt['SimTriggerInfo']['PEList'][:]
    start = ipt['SimTruth/T'][:]

gmu = 160.
gsigma = 40.
p = [8., 0.5, 24.]
p[2] = p[2] * gmu / np.sum(wff.spe(np.arange(window), tau=p[0], sigma=p[1], A=p[2]))
Alpha = 1 / Tau
Co = Alpha / 2. * np.exp(Alpha * Alpha * Sigma * Sigma / 2.)
std = 1.

def start_time(a0, a1):
    stime = np.empty(a1 - a0)
    tlist = torch.arange(window)

    t_auto = tlist[:, None] - tlist
    # amplitude to voltage converter
    AV = p[2] * torch.exp(-1 / 2 * torch.pow((torch.log((t_auto + torch.abs(t_auto)) * (1 / p[0] / 2)) * (1 / p[1])), 2))
    def model(y, mu):
        t0 = pyro.sample('t0', dist.Uniform(0., window))
        pl = Co * (1. - torch.erf((Alpha * Sigma * Sigma - (tlist - t0)) / (np.sqrt(2.) * Sigma))) * torch.exp(-Alpha * (tlist - t0)) * mu

        with pyro.plate('charges', window):
            A = pyro.sample('A', dist.MixtureSameFamily(
                dist.Categorical(torch.stack((1 - pl, pl)).T),
                dist.Normal(torch.tensor((0., 1.)).expand(window, 2), torch.tensor((1/1e3, 1/4)).expand(window, 2))
            ))

        with pyro.plate("observations", window):
            obs = pyro.sample('obs', dist.Normal(0, scale=std), obs=y-torch.matmul(AV, A))
        return obs
    for i in range(a0, a1):
        wave = ent[i]['Waveform']
        nuts_kernel = NUTS(partial(model, mu=np.sum(wave) / gmu), jit_compile=False)
        mcmc = MCMC(nuts_kernel, num_samples=2000, warmup_steps=1000, num_chains=1)
        mcmc.run(torch.from_numpy(wave).float())
        stime[i] = np.mean(mcmc.get_samples()['t0'].numpy())
        mcmc.summary()
    return stime

if args.Ncpu == 1:
    slices = [[0, N]]
else:
    chunk = N // args.Ncpu + 1
    slices = np.vstack((np.arange(0, N, chunk), np.append(np.arange(chunk, N, chunk), N))).T.astype(np.int).tolist()
print('Initialization finished, real time {0:.4f}s, cpu time {1:.4f}s'.format(time.time() - global_start, time.process_time() - cpu_global_start))
tic = time.time()
cpu_tic = time.process_time()

ent = np.sort(ent, kind='stable', order=['TriggerNo', 'ChannelID'])
Chnum = len(np.unique(ent['ChannelID']))
e_pel = ent['TriggerNo'] * Chnum + ent['ChannelID']
e_pel, i_pel = np.unique(e_pel, return_index=True)
i_pel = np.append(i_pel, len(ent))

sdtp = np.dtype([('TriggerNo', np.uint32), ('ChannelID', np.uint32), ('tsfirsttruth', np.float64), ('tsfirstcharge', np.float64), ('tstruth', np.float64), ('tscharge', np.float64)])
ts = np.zeros(N, dtype=sdtp)
ts['TriggerNo'] = np.unique(ent['TriggerNo'])
ts['ChannelID'] = np.unique(ent['ChannelID'])
ts['tscharge'] = np.full(N, np.nan)
ts['tsfirsttruth'] = np.full(N, np.nan)
ts['tsfirstcharge'] = np.full(N, np.nan)

chunk = N // args.Ncpu + 1
slices = np.vstack((np.arange(0, N, chunk), np.append(np.arange(chunk, N, chunk), N))).T.astype(np.int).tolist()
start_time(0, 10)
with Pool(min(args.Ncpu, cpu_count())) as pool:
    result = pool.starmap(partial(start_time, mode='all'), slices)

ts = np.hstack(result)
ts = np.sort(ts, kind='stable', order=['TriggerNo', 'ChannelID'])
print('Prediction generated, real time {0:.4f}s, cpu time {1:.4f}s'.format(time.time() - tic, time.process_time() - cpu_tic))

with h5py.File(args.opt, 'w') as opt:
    dset = opt.create_dataset('risetime', data=ts, compression='gzip')
    dset.attrs['Method'] = method
    dset.attrs['mu'] = Mu
    dset.attrs['tau'] = Tau
    dset.attrs['sigma'] = Sigma
    print('The output file path is {}'.format(args.opt))

print('Finished! Consuming {0:.2f}s in total, cpu time {1:.2f}s.'.format(time.time() - global_start, time.process_time() - cpu_global_start))
