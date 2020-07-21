# -*- coding: utf-8 -*-

import sys
import re
import numpy as np
import h5py
import math
import argparse
import pickle
from functools import partial
import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from multiprocessing import Pool
import wf_func as wff

Demo = False

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', type=str, help='output file')
psr.add_argument('ipt', type=str, help='input file')
psr.add_argument('--mod', type=str, help='mode of weight or charge')
psr.add_argument('--met', type=str, help='fitting method')
psr.add_argument('--ref', type=str, nargs='+', help='reference file')
psr.add_argument('-N', dest='Ncpu', type=int, help='cpu number', default=50)
psr.add_argument('--demo', dest='demo', action='store_true', help='demo bool', default=False)
args = psr.parse_args()

fipt = args.ipt
fopt = args.opt
reference = args.ref
mode = args.mod
method = args.met
Ncpu = args.Ncpu
if args.demo:
    Demo = True

Thres = 0.1
warmup = 200
samples = 1000
E = 0

def lasso_select(pf_r, wave, mne):
    pf_r = pf_r[np.argmin([loss(pf_r[j], mne, wave) for j in range(len(pf_r))])]
    return pf_r

def loss(x, M, y, eta=E):
    return np.power(y - np.matmul(M, x), 2).sum() + eta * x.sum()

def model(wave, mne, n, eta=E):
    pf = numpyro.sample('weight', dist.HalfNormal(jnp.ones(n)))
    y = numpyro.sample('y', dist.Normal(0, 1), obs=jnp.power(wave-jnp.matmul(mne, pf), 2) + eta*jnp.sum(pf))
    return y

def inferencing(a, b):
    spe = np.vstack([np.concatenate((spe_pre[i]['spe'], np.zeros(leng - len(spe_pre[i]['spe'])))) for i in spe_pre.keys()])

    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    model_collect = {}
    nuts_kernel_collect = {}
    mcmc_collect = {}
    with h5py.File(fipt, 'r', libver='latest', swmr=True) as ipt:
        ent = ipt['Waveform']
        dt = np.zeros((b - a) * (leng//5), dtype=opdt)
        start = 0
        end = 0
        for i in range(a, b):
            cid = ent[i]['ChannelID']
            wave = wff.deduct_base(spe_pre[cid]['epulse'] * ent[i]['Waveform'], spe_pre[cid]['m_l'], spe_pre[cid]['thres'], 20, 'detail')
            pos = np.argwhere(wave[spe_pre[cid]['peak_c'] + 2:] > spe_pre[cid]['thres']).flatten()
            pf = wave[pos]/(spe_pre[cid]['spe'].sum())
            flag = 1
            if len(pos) == 0:
                flag = 0
            else:
                mne = spe[cid][np.mod(np.arange(leng).reshape(leng, 1) - pos.reshape(1, len(pos)), leng)]

                # op = stanmodel.sampling(data=dict(m=mne, y=wave, Nf=leng, Np=len(pos)), iter=1000, seed=0)
                # pf = lasso_select(op['x'], wave, mne)
                if not len(pos) in mcmc_collect:
                    model_collect.update({len(pos) : partial(model, n=len(pos), eta=E)})
                    nuts_kernel_collect.update({len(pos) : NUTS(model_collect[len(pos)], step_size=0.01, adapt_step_size=True)})
                    mcmc_collect.update({len(pos) : MCMC(nuts_kernel_collect[len(pos)], num_warmup=warmup, num_samples=samples, num_chains=1, progress_bar=Demo, jit_model_args=True)})
                mcmc_collect[len(pos)].run(rng_key, wave=jnp.array(wave), mne=jnp.array(mne))
                # pf = np.mean(np.array(mcmc_collect[len(pos)].get_samples()['weight']), axis=0)
                pf = lasso_select(np.array(mcmc_collect[len(pos)].get_samples()['weight']), wave, mne)
                pos_r = pos[pf > Thres]
                pf = pf[pf > Thres]
                if len(pos_r) == 0:
                    flag = 0
            if flag == 0:
                t = np.array([np.argmax(wave)]) - spe_pre[cid]['peak_c']
                pf = np.array([1])
                pos_r = t if t[0] >= 0 else np.array([0])
            lenpf = len(pf)
            end = start + lenpf
            dti['PETime'][start:end] = pos_r.astype(np.uint16)
            if mode == 'Weight':
                dt[mode][start:end] = pf.astype(np.float16)
            elif mode == 'Charge':
                dt[mode][start:end] = pf.astype(np.float16) * np.sum(spe[cid])
            dt['EventID'][start:end] = ent[i]['EventID']
            dt['ChannelID'][start:end] = ent[i]['ChannelID']
            start = end
    dt = dt[dt[mode] > 0]
    dt = np.sort(dt, kind='stable', order=['EventID', 'ChannelID', 'PETime'])
    return dt

def fitting(a, b):
    with h5py.File(fipt, 'r', libver='latest', swmr=True) as ipt:
        ent = ipt['Waveform']
        leng = len(ent[0]['Waveform'])
        dt = np.zeros((b - a) * (leng//5), dtype=opdt)
        start = 0
        end = 0
        for i in range(a, b):
            wave = wff.deduct_base(spe_pre[ent[i]['ChannelID']]['epulse'] * ent[i]['Waveform'], spe_pre[ent[i]['ChannelID']]['m_l'], spe_pre[ent[i]['ChannelID']]['thres'], 20, 'detail')

            if method == 'xiaopeip':
                pf = wff.fit_N(wave, spe_pre[ent[i]['ChannelID']], 'xiaopeip', eta=E)
            elif method == 'lucyddm':
                pf = wff.lucyddm_core(wave, spe_pre[ent[i]['ChannelID']]['spe'])
            pet, pwe = wff.pf_to_tw(pf, Thres)

            lenpf = len(pwe)
            end = start + lenpf
            dt['PETime'][start:end] = pet.astype(np.uint16)
            if mode == 'Weight':
                dt[mode][start:end] = pwe.astype(np.float16)
            elif mode == 'Charge':
                dt[mode][start:end] = pwe.astype(np.float16) * np.sum(spe_pre[ent[i]['ChannelID']]['spe'])
            dt['EventID'][start:end] = ent[i]['EventID']
            dt['ChannelID'][start:end] = ent[i]['ChannelID']
            start = end
    dt = dt[dt[mode] > 0]
    dt = np.sort(dt, kind='stable', order=['EventID', 'ChannelID', 'PETime'])
    return dt

spe_pre = wff.read_model(reference[0])
stanmodel = pickle.load(open(reference[1], 'rb'))
opdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint32), ('PETime', np.uint16), (mode, np.float16)])
with h5py.File(fipt, 'r', libver='latest', swmr=True) as ipt:
    l = len(ipt['Waveform'])
    print('{} waveforms will be computed'.format(l))
    leng = len(ipt['Waveform'][0]['Waveform'])
    assert leng >= len(spe_pre[0]['spe']), 'Single PE too long which is {}'.format(len(spe_pre[0]['spe']))
chunk = l // Ncpu + 1
slices = np.vstack((np.arange(0, l, chunk), np.append(np.arange(chunk, l, chunk), l))).T.astype(np.int).tolist()
with Pool(Ncpu) as pool:
    if method == 'mcmc':
        select_result = pool.starmap(inferencing, slices)
    else:
        select_result = pool.starmap(fitting, slices)
#select_result = inferencing(slices[0][0], slices[-1][-1])
result = np.hstack(select_result)
with h5py.File(fopt, 'w') as opt:
    dset = opt.create_dataset('Answer', data=result, compression='gzip')
    dset.attrs['Method'] = method
    print('The output file path is {}'.format(fopt))
