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
import scipy
import pandas as pd
from tqdm import tqdm
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpyro
numpyro.set_platform('cpu')
# numpyro.set_host_device_count(2)
# import numpyro.contrib.tfp.distributions
import matplotlib.pyplot as plt

import wf_func as wff

global_start = time.time()
cpu_global_start = time.process_time()

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', type=str, help='output file')
psr.add_argument('ipt', type=str, help='input file')
psr.add_argument('--met', type=str, help='fitting method')
psr.add_argument('--ref', type=str, help='reference file')
psr.add_argument('-N', '--Ncpu', dest='Ncpu', type=int, default=25)
args = psr.parse_args()

fipt = args.ipt
fopt = args.opt
reference = args.ref
method = args.met
Demo = False

spe_pre = wff.read_model(reference)
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
    simtruth = ipt['SimTruth/T'][:]

gmu = 160.
gsigma = 40.
p = spe_pre[0]['parameters']
if Tau != 0:
    Alpha = 1 / Tau
    Co = (Alpha / 2. * np.exp(Alpha ** 2 * Sigma ** 2 / 2.)).item()
std = 1.
Thres = 0.2

class mNormal(numpyro.distributions.distribution.Distribution):
    arg_constraints = {'pl': numpyro.distributions.constraints.real}
    # support = numpyro.distributions.constraints.real
    support = numpyro.distributions.constraints.positive
    reparametrized_params = ['pl']

    def __init__(self, pl, s, mu, sigma, validate_args=None):
        self.pl = pl
        self.s = s
        self.mu = mu
        self.sigma = sigma
        self.norm0 = numpyro.distributions.Normal(loc=0., scale=self.s)
        self.norm1 = numpyro.distributions.Normal(loc=self.mu, scale=self.sigma)
        super(mNormal, self).__init__(batch_shape=jnp.shape(pl), validate_args=validate_args)

    @numpyro.distributions.util.validate_sample
    def log_prob(self, value):
        logprob0 = self.norm0.log_prob(value)
        logprob1 = self.norm1.log_prob(value)
        prob = jnp.vstack([logprob0, logprob1])
        pl = jnp.vstack([(1 - self.pl), self.pl])
        return jax.scipy.special.logsumexp(prob, axis=0, b=pl) + jnp.log(2)

def time_numpyro(a0, a1):
    npe = 6
    Awindow = int(window * 0.95)
    rng_key = jax.random.PRNGKey(1)
    rng_key, rng_key_ = jax.random.split(rng_key)
    stime = np.empty(a1 - a0)
    accep = np.full(a1 - a0, np.nan)
    dt = np.zeros((a1 - a0) * Awindow * 2, dtype=opdt)
    start = 0
    end = 0
    count = 0
    def model(n, y, mu, tlist, AV, t0right, t0left):
        t0 = numpyro.sample('t0', numpyro.distributions.Uniform(t0right, t0left))
        if Tau == 0:
            light_curve = numpyro.distributions.Normal(t0, scale=Sigma)
            pl = numpyro.primitives.deterministic('pl', jnp.exp(light_curve.log_prob(tlist)) * mu / n + jnp.finfo(jnp.float64).epsneg)
            # pl = numpyro.primitives.deterministic('pl', 1 / (math.sqrt(2. * math.pi) * Sigma) * jnp.exp(-((tlist - t0) / Sigma) ** 2 / 2) * mu / n + jnp.finfo(jnp.float64).epsneg)
        else:
            pl = numpyro.primitives.deterministic('pl', Co * (1. - jax.scipy.special.erf((Alpha * Sigma ** 2 - (tlist - t0)) / (math.sqrt(2.) * Sigma))) * jnp.exp(-Alpha * (tlist - t0)) * mu / n + jnp.finfo(jnp.float64).epsneg)
        # A = numpyro.sample('A', numpyro.contrib.tfp.distributions.MixtureSameFamily(
        #     numpyro.contrib.tfp.distributions.Categorical(jnp.vstack([(1 - pl), pl]).T),
        #     numpyro.contrib.tfp.distributions.Normal(jnp.repeat(jnp.array([[0., 1.]]), window, axis=0), jnp.repeat(jnp.array([[std / gsigma, gsigma / gmu]]), window, axis=0))
        # ))
        A = numpyro.sample('A', mNormal(pl, std / gsigma, 1., gsigma / gmu))
        with numpyro.plate('observations', window):
            obs = numpyro.sample('obs', numpyro.distributions.Normal(jnp.matmul(AV, A), scale=std), obs=y)
        return obs
    for i in range(a0, a1):
        petime = pelist[pelist['TriggerNo'] == ent[i]['TriggerNo']]
        cid = ent[i]['ChannelID']
        wave = jnp.array(ent[i]['Waveform'].astype(np.float32)) * spe_pre[cid]['epulse']
        mu = jnp.sum(wave) / gmu
        n = max(round(mu / math.sqrt(Tau ** 2 + Sigma ** 2)), 1)
        hitt, char = wff.lucyddm(ent[i]['Waveform'], spe_pre[cid])
        hitt, char = wff.clip(hitt, char, Thres)
        char = char / char.sum() * jnp.clip(jnp.abs(wave.sum()), 1e-6, jnp.inf) / spe_pre[cid]['spe'].sum()
        t0_init = jnp.array(wff.likelihoodt0(hitt=hitt, char=char, gmu=gmu, gsigma=gsigma, Tau=Tau, Sigma=Sigma, npe=npe, mode='charge'))
        right = jnp.clip(hitt.min() - round(3 * spe_pre[cid]['mar_l']), 0, window)
        left = jnp.clip(hitt.max() + round(3 * spe_pre[cid]['mar_l']), 0, window)
        tlist = jnp.arange(right, left, 1 / n)
        A_init = np.zeros(left - right)
        A_init[hitt - hitt.min() + round(3 * spe_pre[cid]['mar_l'])] = char
        A_init = jnp.repeat(A_init, n) / n + jnp.finfo(jnp.float64).epsneg
        t_auto = jnp.arange(window)[:, None] - tlist
        AV = p[2] * jnp.exp(-1 / 2 * jnp.power((jnp.log((t_auto + jnp.abs(t_auto)) * (1 / p[0] / 2)) * (1 / p[1])), 2))
        nuts_kernel = numpyro.infer.NUTS(model, adapt_step_size=True, init_strategy=numpyro.infer.initialization.init_to_value(values={'t0': t0_init, 'A': A_init}))
        mcmc = numpyro.infer.MCMC(nuts_kernel, num_samples=1000, num_warmup=300, num_chains=1, progress_bar=Demo, chain_method='sequential', jit_model_args=True)
        try:
            ticrun = time.time()
            mcmc.run(rng_key, n=n, y=wave, mu=mu, tlist=tlist, AV=AV, t0right=t0_init - 3 * Sigma, t0left=t0_init + 3 * Sigma, extra_fields=('num_steps', 'accept_prob'))
            tocrun = time.time()
            num_leapfrogs = mcmc.get_extra_fields()['num_steps'].sum()
            # print('avg. time for each step :', (tocrun - ticrun) / num_leapfrogs)
            accep[i - a0] = np.array(mcmc.get_extra_fields()['accept_prob']).mean()
            t0 = np.array(mcmc.get_samples()['t0']).flatten()
            A = np.array(mcmc.get_samples()['A'])
            if np.std(t0) < 2 * Sigma:
                count = count + 1
            else:
                raise ValueError
        except:
            t0 = np.array([t0_init])
            tlist = hitt
            A = np.array([char])
            print('Failed waveform is TriggerNo = {:05d}, ChannelID = {:02d}, i = {:05d}'.format(ent[i]['TriggerNo'], ent[i]['ChannelID'], i))
        stime[i - a0] = np.mean(t0)
        pet, cha = wff.clip(np.array(tlist), np.mean(A, axis=0), Thres)
        end = start + len(cha)
        dt['HitPosInWindow'][start:end] = pet
        cha = cha / cha.sum() * np.clip(np.abs(wave.sum()), 1e-6, np.inf)
        dt['Charge'][start:end] = cha
        dt['TriggerNo'][start:end] = ent[i]['TriggerNo']
        dt['ChannelID'][start:end] = ent[i]['ChannelID']
        start = end
    dt = dt[:end]
    dt = np.sort(dt, kind='stable', order=['TriggerNo', 'ChannelID'])
    return stime, dt, count, accep

if args.Ncpu == 1:
    slices = [[0, N]]
else:
    chunk = N // args.Ncpu + 1
    slices = np.vstack((np.arange(0, N, chunk), np.append(np.arange(chunk, N, chunk), N))).T.astype(int).tolist()

print('Initialization finished, real time {0:.02f}s, cpu time {1:.02f}s'.format(time.time() - global_start, time.process_time() - cpu_global_start))
tic = time.time()
cpu_tic = time.process_time()

ent = np.sort(ent, kind='stable', order=['TriggerNo', 'ChannelID'])
Chnum = len(np.unique(ent['ChannelID']))
e_pel = ent['TriggerNo'] * Chnum + ent['ChannelID']
e_pel, i_pel = np.unique(e_pel, return_index=True)
i_pel = np.append(i_pel, len(ent))

sdtp = np.dtype([('TriggerNo', np.uint32), ('ChannelID', np.uint32), ('tswave', np.float64)])
ts = np.zeros(N, dtype=sdtp)
ts['TriggerNo'] = ent['TriggerNo']
ts['ChannelID'] = ent['ChannelID']
pelist = np.sort(pelist, kind='stable', order=['TriggerNo', 'PMTId', 'HitPosInWindow'])
Chnum = len(np.unique(ent['ChannelID']))
e_pel = pelist['TriggerNo'] * Chnum + pelist['PMTId']
e_pel, i_pel = np.unique(e_pel, return_index=True)
i_pel = np.append(i_pel, len(pelist))
opdt = np.dtype([('TriggerNo', np.uint32), ('ChannelID', np.uint32), ('HitPosInWindow', np.float64), ('Charge', np.float64)])

with Pool(min(args.Ncpu, cpu_count())) as pool:
    result = pool.starmap(partial(time_numpyro), slices)
ts['tswave'] = np.hstack([result[i][0] for i in range(len(slices))])
As = np.hstack([result[i][1] for i in range(len(slices))])
count = np.sum([result[i][2] for i in range(len(slices))])
accep = np.hstack([result[i][3] for i in range(len(slices))])

ff = plt.figure(figsize=(8, 6))
ax = ff.add_subplot()
ax.hist(accep, bins=np.arange(0, 1+0.02, 0.02), label='accept_prob')
ax.legend(loc='upper right')
ff.savefig(os.path.splitext(fopt)[0] + '.png')
plt.close()

As = np.sort(As, kind='stable', order=['TriggerNo', 'ChannelID'])
print('Successful MCMC ratio is {:.4%}'.format(count / N))
print('Prediction generated, real time {0:.02f}s, cpu time {1:.02f}s'.format(time.time() - tic, time.process_time() - cpu_tic))

with h5py.File(fopt, 'w') as opt:
    pedset = opt.create_dataset('photoelectron', data=As, compression='gzip')
    pedset.attrs['Method'] = method
    pedset.attrs['mu'] = Mu
    pedset.attrs['tau'] = Tau
    pedset.attrs['sigma'] = Sigma
    tsdset = opt.create_dataset('starttime', data=ts, compression='gzip')
    print('The output file path is {}'.format(fopt))

print('Finished! Consuming {0:.02f}s in total, cpu time {1:.02f}s.'.format(time.time() - global_start, time.process_time() - cpu_global_start))