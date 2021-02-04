# -*- coding: utf-8 -*-

import sys
import re
import time
import math
import argparse
import pickle
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
# np.seterr(all='raise')
import h5py
import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from tqdm import tqdm

import wf_func as wff

global_start = time.time()
cpu_global_start = time.process_time()

Demo = False

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', type=str, help='output file')
psr.add_argument('ipt', type=str, help='input file')
psr.add_argument('--met', type=str, help='fitting method')
psr.add_argument('--ref', type=str, nargs='+', help='reference file')
psr.add_argument('-N', '--Ncpu', dest='Ncpu', type=int, default=50)
psr.add_argument('--demo', dest='demo', action='store_true', help='demo bool', default=False)
args = psr.parse_args()

fipt = args.ipt
fopt = args.opt
reference = args.ref
method = args.met
if args.demo:
    Demo = True

Thres = 0.1
warmup = 200
samples = 1000
E = 0.

def model(wave, mne, n, eta=E):
    pf = numpyro.sample('penum', dist.HalfNormal(jnp.ones(n)))
    y = numpyro.sample('y', dist.Normal(0, 1), obs=jnp.power(wave-jnp.matmul(mne, pf), 2) + eta*jnp.sum(pf))
    return y

def inferencing(a, b):
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    model_collect = {}
    nuts_kernel_collect = {}
    mcmc_collect = {}
    tlist = np.arange(window)
    with h5py.File(fipt, 'r', libver='latest', swmr=True) as ipt:
        ent = ipt['Readout/Waveform']
        spe = np.vstack([np.concatenate((spe_pre[i]['spe'], np.zeros(window - len(spe_pre[i]['spe'])))) for i in spe_pre.keys()])
        dt = np.zeros((b - a) * (window//5), dtype=opdt)
        start = 0
        end = 0
        for i in range(a, b):
            cid = ent[i]['ChannelID']
            wave = ent[i]['Waveform'].astype(np.float) * spe_pre[ent[i]['ChannelID']]['epulse']
            pos = np.argwhere(wave[spe_pre[cid]['peak_c'] + 2:] > 5 * spe_pre[cid]['std']).flatten()
            pwe = wave[pos]/(spe_pre[cid]['spe'].sum())
            flag = 1
            if len(pos) != 0:
                mne = spe[cid][np.mod(tlist.reshape(window, 1) - pos.reshape(1, len(pos)), window)]
                # op = stanmodel.sampling(data=dict(m=mne, y=wave, Nf=window, Np=len(pos)), iter=1000, seed=0)
                # pwe = lasso_select(op['x'], wave, mne)
                if not len(pos) in mcmc_collect:
                    model_collect.update({len(pos) : partial(model, n=len(pos), eta=E)})
                    nuts_kernel_collect.update({len(pos) : NUTS(model_collect[len(pos)], step_size=0.01, adapt_step_size=True)})
                    mcmc_collect.update({len(pos) : MCMC(nuts_kernel_collect[len(pos)], num_warmup=warmup, num_samples=samples, num_chains=1, progress_bar=Demo, jit_model_args=True)})
                mcmc_collect[len(pos)].run(rng_key, wave=jnp.array(wave), mne=jnp.array(mne))
                pwe = np.mean(np.array(mcmc_collect[len(pos)].get_samples()['penum']), axis=0)
                # pwe = lasso_select(np.array(mcmc_collect[len(pos)].get_samples()['penum']), wave, mne)
            pet, pwe = wff.clip(pet, pwe, Thres)
            end = start + len(pet)
            dti['HitPosInWindow'][start:end] = pet
            pwe = pwe / pwe.sum() * np.clip(np.abs(wave.sum()), 1e-6, np.inf)
            dt['Charge'][start:end] = pwe
            dt['TriggerNo'][start:end] = ent[i]['TriggerNo']
            dt['ChannelID'][start:end] = ent[i]['ChannelID']
            start = end
    dt = dt[:end]
    dt = np.sort(dt, kind='stable', order=['TriggerNo', 'ChannelID'])
    return dt

def fitting(a, b):
    with h5py.File(fipt, 'r', libver='latest', swmr=True) as ipt:
        ent = ipt['Readout/Waveform'][:]
        dt = np.zeros((b - a) * window, dtype=opdt)
        start = 0
        end = 0
        for i in range(a, b):
            wave = ent[i]['Waveform'].astype(np.float) * spe_pre[ent[i]['ChannelID']]['epulse']

            if method == 'xiaopeip':
#                 pet, pwe, ped = wff.xiaopeip(wave, spe_pre[ent[i]['ChannelID']])
#                 wave = wave - ped
                pet, pwe = wff.xiaopeip(wave, spe_pre[ent[i]['ChannelID']])
            elif method == 'lucyddm':
                pet, pwe = wff.lucyddm(wave, spe_pre[ent[i]['ChannelID']]['spe'], iterations=50)
            elif method == 'threshold':
                pet, pwe = wff.threshold(wave, spe_pre[ent[i]['ChannelID']])
            elif method == 'fftrans':
                pet, pwe = wff.waveformfft(wave, spe_pre[ent[i]['ChannelID']])
            elif method == 'findpeak':
                pet, pwe = wff.findpeak(wave, spe_pre[ent[i]['ChannelID']])
            pet, pwe = wff.clip(pet, pwe, Thres)

            end = start + len(pwe)
            dt['HitPosInWindow'][start:end] = pet
            pwe = pwe / pwe.sum() * np.clip(np.abs(wave.sum()), 1e-6, np.inf)
            dt['Charge'][start:end] = pwe
            dt['TriggerNo'][start:end] = ent[i]['TriggerNo']
            dt['ChannelID'][start:end] = ent[i]['ChannelID']
            start = end
    dt = dt[:end]
    dt = np.sort(dt, kind='stable', order=['TriggerNo', 'ChannelID'])
    return dt

spe_pre = wff.read_model(reference[0])
opdt = np.dtype([('TriggerNo', np.uint32), ('ChannelID', np.uint32), ('HitPosInWindow', np.uint16), ('Charge', np.float64)])
with h5py.File(fipt, 'r', libver='latest', swmr=True) as ipt:
    l = len(ipt['Readout/Waveform'])
    print('{} waveforms will be computed'.format(l))
    window = len(ipt['Readout/Waveform'][0]['Waveform'])
    assert window >= len(spe_pre[0]['spe']), 'Single PE too long which is {}'.format(len(spe_pre[0]['spe']))
    Mu = ipt['Readout/Waveform'].attrs['mu']
    Tau = ipt['Readout/Waveform'].attrs['tau']
    Sigma = ipt['Readout/Waveform'].attrs['sigma']
if args.Ncpu == 1:
    slices = [[0, l]]
else:
    chunk = l // args.Ncpu + 1
    slices = np.vstack((np.arange(0, l, chunk), np.append(np.arange(chunk, l, chunk), l))).T.astype(np.int).tolist()
print('Initialization finished, real time {0:.02f}s, cpu time {1:.02f}s'.format(time.time() - global_start, time.process_time() - cpu_global_start))
tic = time.time()
cpu_tic = time.process_time()
with Pool(min(args.Ncpu, cpu_count())) as pool:
    if method == 'mcmc':
        select_result = pool.starmap(inferencing, slices)
    else:
        select_result = pool.starmap(fitting, slices)
result = np.hstack(select_result)
result = np.sort(result, kind='stable', order=['TriggerNo', 'ChannelID'])
print('Prediction generated, real time {0:.02f}s, cpu time {1:.02f}s'.format(time.time() - tic, time.process_time() - cpu_tic))
with h5py.File(fopt, 'w') as opt:
    dset = opt.create_dataset('photoelectron', data=result, compression='gzip')
    dset.attrs['Method'] = method
    dset.attrs['mu'] = Mu
    dset.attrs['tau'] = Tau
    dset.attrs['sigma'] = Sigma
    print('The output file path is {}'.format(fopt))
print('Finished! Consuming {0:.02f}s in total, cpu time {1:.02f}s.'.format(time.time() - global_start, time.process_time() - cpu_global_start))