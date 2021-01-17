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
import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
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
lgmu = np.log(gmu) - 0.5 * np.log(1 + gsigma**2 / gmu**2)
lgsigma = np.log(1 + gsigma**2 / gmu**2)
p = [8., 0.5, 24.]
p[2] = p[2] * gmu / np.sum(wff.spe(np.arange(window), tau=p[0], sigma=p[1], A=p[2]))
std = 1.

def start_time_numpyro(a0, a1):
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    stime = np.empty(a1 - a0)
    tlist = jnp.arange(window)
    def model(y):
        n = 2
        t0 = numpyro.sample('t0', dist.Uniform(50., 550.))
        tb = numpyro.sample('tb', dist.Normal(jnp.ones(n) * t0, scale=Sigma))
        if Tau != 0:
            ta = numpyro.sample('ta', dist.Exponential(rate=jnp.ones(n) * 1/Tau))
            t = ta + tb
        else:
            t = tb
        A = numpyro.sample('A', dist.LogNormal(jnp.ones(n) * lgmu, scale=lgsigma))
        obs = numpyro.sample('obs', dist.Normal(0, scale=std), obs=y-jnp.sum(p[2] * jnp.exp(-1 / 2 * jnp.power((jnp.log(((tlist - t[:, None] + jnp.abs(tlist - t[:, None]))/2) / p[0]) / p[1]), 2)) * A[:, None] / gmu, axis=0))
        return obs
    nuts_kernel = NUTS(model, adapt_step_size=True)
    mcmc = MCMC(nuts_kernel, num_warmup=2000, num_samples=2000, num_chains=1)
    for i in range(a0, a1):
        mcmc.run(rng_key, y=jnp.array(ent[i]['Waveform']))
        stime[i] = np.mean(np.array(mcmc.get_samples()['t0']))
        mcmc.print_summary()
    return stime

def start_time(a0, a1):
    stime = np.empty(a1 - a0)
    tlist = torch.arange(window)
    def model(y):
        n = 2
        t0 = pyro.sample('t0', dist.Uniform(50., 550.))
        tb = pyro.sample('tb', dist.Normal(torch.ones(n) * t0, scale=Sigma))
        if Tau != 0:
            ta = pyro.sample('ta', dist.Exponential(rate=torch.ones(n) * 1/Tau))
            t = ta + tb
        else:
            t = tb
        A = pyro.sample('A', dist.LogNormal(torch.ones(n) * lgmu, scale=lgsigma))
        wave = torch.sum(p[2] * torch.exp(-1 / 2 * torch.pow((torch.log(((tlist - t[:, None] + torch.abs(tlist - t[:, None]))/2) / p[0]) / p[1]), 2)) * A[:, None] / gmu, axis=0)
        obs = pyro.sample('obs', dist.Normal(wave, scale=std), obs=y)
        return obs
    nuts_kernel = NUTS(model, jit_compile=False)
    mcmc = MCMC(nuts_kernel, num_samples=2000, warmup_steps=2000, num_chains=1)
    for i in range(a0, a1):
        mcmc.run(torch.from_numpy(ent[i]['Waveform']).float())
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
start_time_numpyro(0, 10)
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