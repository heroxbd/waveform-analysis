# -*- coding: utf-8 -*-

import os
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
torch.manual_seed(0)
# torch.autograd.set_detect_anomaly(True)
import torch.distributions.constraints as constraints
from torch.distributions.utils import broadcast_all
import pyro
import pyro.distributions as dist
from torch.distributions.exp_family import ExponentialFamily
from pyro.distributions.torch import Normal
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.torch_distribution import TorchDistributionMixin
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc import NUTS, HMC
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

use_cuda = False

if use_cuda:
    device = torch.device(0)
    torch.cuda.init()
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')

spe_pre = wff.read_model(reference[0])
with h5py.File(fipt, 'r', libver='latest', swmr=True) as ipt:
    ent = ipt['Readout/Waveform'][:]
    N = len(ent)
    print('{} waveforms will be computed'.format(N))
    window = len(ent[0]['Waveform'])
    assert window >= len(spe_pre[0]['spe']), 'Single PE too long which is {}'.format(len(spe_pre[0]['spe']))
    Mu = ipt['Readout/Waveform'].attrs['mu'].item()
    Tau = ipt['Readout/Waveform'].attrs['tau'].item()
    Sigma = ipt['Readout/Waveform'].attrs['sigma'].item()
    pelist = ipt['SimTriggerInfo']['PEList'][:]
    start = ipt['SimTruth/T'][:]

gmu = 160.
gsigma = 40.
p = [8., 0.5, 24.]
p[2] = (p[2] * gmu / np.sum(wff.spe(np.arange(window), tau=p[0], sigma=p[1], A=p[2]))).item()
if Tau != 0:
    Alpha = 1 / Tau
    Co = (Alpha / 2. * np.exp(Alpha * Alpha * Sigma * Sigma / 2.)).item()
std = 1.

class mNormal(ExponentialFamily, TorchDistribution):
    has_rsample = True
    support = constraints.real
    def __init__(self, pl, s, mu, sigma, validate_args=None):
        self.pl, self.s, self.mu, self.sigma = broadcast_all(pl + torch.finfo(pl.dtype).tiny, s, mu, sigma)
        batch_shape = self.pl.size()
        # self.a = 1 / (2 * self.s ** 2) - 1 / (2 * self.sigma ** 2)
        # self.b = self.mu / (self.sigma ** 2)
        # self.c = -self.mu ** 2 / (2 * self.sigma ** 2) + torch.log(self.pl * self.s / (1 - self.pl) / self.sigma)
        # self.intersect = ((-self.b + torch.sqrt(self.b ** 2 - 4 * self.a * self.c)) / (2 * self.a)).detach()
        self.intersect = torch.ones(self.pl.shape, dtype=self.pl.dtype, device=self.pl.device).detach() * 0.1
        self.norm0 = Normal(loc=torch.zeros(self.pl.shape, dtype=self.pl.dtype, device=self.pl.device), scale=self.s)
        self.norm1 = Normal(loc=self.mu, scale=self.sigma)
        # self.rpl = 1 - (1 - self.pl) * self.norm0.cdf(self.intersect)
        self.rpl = self.pl
        super(mNormal, self).__init__(batch_shape, validate_args=validate_args)

    # def sample(self, sample_shape=torch.Size()):
    #     shape = self._extended_shape(sample_shape)
    #     m = dist.Bernoulli(self.pl)
    #     mix = m.sample()
    #     with torch.no_grad():
    #         return torch.where(mix > 0., torch.normal(self.mu.expand(shape), self.sigma.expand(shape)), torch.normal(torch.zeros(shape), self.s.expand(shape)))
    
    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return self.rsample(sample_shape=shape)
    
    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape, dtype=self.pl.dtype, device=self.pl.device)
        return torch.where(u < 1 - self.rpl, self.s * torch.erfinv(2 * self.norm0.cdf(self.intersect) / (1 - self.rpl) * u - 1) * math.sqrt(2), self.mu + self.sigma * torch.erfinv(2 * (self.norm1.cdf(self.intersect) - 1) / (-self.rpl) * (u - 1) + 1) * math.sqrt(2))

    def log_prob(self, value):
        return torch.where(value < self.intersect, (1 - self.pl).log() + self.norm0.log_prob(value), self.pl.log() + self.norm1.log_prob(value))

def start_time(a0, a1):
    stime = np.empty(a1 - a0)
    tlist = torch.arange(window).to(device)
    t_auto = (tlist[:, None] - tlist).to(device)
    # amplitude to voltage converter
    AV = (p[2] * torch.exp(-1 / 2 * torch.pow((torch.log((t_auto + torch.abs(t_auto)) * (1 / p[0] / 2)) * (1 / p[1])), 2))).to(device)
    Amu = torch.tensor((0., 1.)).expand(window, 2).to(device)
    Asigma = torch.tensor((std / gsigma, gsigma / gmu)).expand(window, 2).to(device)
    wmu = torch.tensor(0.).to(device)
    wsigma = torch.tensor(std).to(device)
    left = torch.tensor(0.).to(device)
    right = torch.tensor(float(window)).to(device)
    def model(y, mu):
        t0 = pyro.sample('t0', dist.Uniform(left, right)).to(device)
        if Tau == 0:
            light_curve = dist.Normal(t0, scale=Sigma)
            pl = (torch.exp(light_curve.log_prob(tlist)) * mu).to(device)
        else:
            pl = (Co * (1. - torch.erf((Alpha * Sigma * Sigma - (tlist - t0)) / (math.sqrt(2.) * Sigma))) * torch.exp(-Alpha * (tlist - t0)) * mu).to(device)

        A = pyro.sample('A', dist.MixtureSameFamily(
            dist.Categorical(torch.stack((1 - pl, pl)).T.to(device)),
            dist.Normal(Amu, Asigma)
        ))
        # A = pyro.sample('A', mNormal(pl, torch.tensor(std / gsigma).to(device), torch.tensor(1.).to(device), torch.tensor(gsigma / gmu).to(device)))

        with pyro.plate('observations', window):
            obs = pyro.sample('obs', dist.Normal(wmu, scale=wsigma), obs=y-torch.matmul(AV, A)).to(device)
        return obs
    for i in range(a0, a1):
        wave = torch.from_numpy(ent[i]['Waveform'].astype(np.float32)).to(device)
        nuts_kernel = NUTS(partial(model, mu=1 / gmu * torch.sum(wave)), jit_compile=False)
        mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=500, num_chains=1)
        mcmc.run(wave)
        stime[i] = np.mean(mcmc.get_samples()['t0'].cpu().numpy())
    return stime

if args.Ncpu == 1:
    slices = [[0, N]]
else:
    chunk = N // args.Ncpu + 1
    slices = np.vstack((np.arange(0, N, chunk), np.append(np.arange(chunk, N, chunk), N))).T.astype(np.int).tolist()
if not os.path.exists('stanmodel.pkl'):
    os.system('python3 stanmodel.py')
stanmodel = pickle.load(open('stanmodel.pkl', 'rb'))

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
start_time(0, 1)
with Pool(min(args.Ncpu, cpu_count())) as pool:
    result = pool.starmap(partial(start_time), slices)

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