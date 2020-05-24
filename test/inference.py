# -*- coding: utf-8 -*-

import sys
import re
import h5py
import math
import numpy as np
from tqdm import tqdm
from scipy import optimize as opti
from functools import partial
import argparse
import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import initialize_model
from numpyro.util import set_platform 
set_platform(platform='cpu')

import wf_func as wff

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', type=str, help='output file')
psr.add_argument('ipt', type=str, help='input file')
psr.add_argument('--met', type=str, help='fitting method')
psr.add_argument('--ref', type=str, help='reference file')
psr.add_argument('--num', type=int, help='fragment number')
psr.add_argument('--demo', dest='demo', action='store_true', help='demo bool', default=False)
args = psr.parse_args()

Demo = args.demo
thres = 0.05
warmup = 200
samples = 1000
E = 0

def lasso_select(pf_r, wave, mne, eta=0):
    pf_r = pf_r[np.argmin([loss(pf_r[j], mne, wave, eta) for j in range(len(pf_r))])]
    return pf_r

def loss(x, M, y, eta=0):
    return np.power(y - np.matmul(M, x), 2).sum() + eta * x.sum()

def model(wave, mne, n, eta):
    pf = numpyro.sample('weight', dist.HalfNormal(jnp.ones(n)))
    y = numpyro.sample('y', dist.Normal(0, 1), obs=jnp.power(wave-jnp.matmul(mne, pf), 2) + eta*jnp.sum(pf))
    return y

def main(fopt, fipt, reference):
    spe_pre = wff.read_model(reference)
    opdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint32), ('PETime', np.uint16), ('Weight', np.float16)])
    ipt = h5py.File(fipt, 'r', libver='latest', swmr=True)
    ent = ipt['Waveform']
    Chnum = np.unique(ent['ChannelID'])
    leng = len(ent[0]['Waveform'])
    assert leng >= len(spe_pre[1]['spe']), 'Single PE too long which is {}'.format(len(spe_pre[1]['spe']))
    spe = np.vstack([np.concatenate((spe_pre[i]['spe'], np.zeros(leng - len(spe_pre[i]['spe'])))) for i in Chnum])

    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    model_collect = {}
    nuts_kernel_collect = {}
    mcmc_collect = {}

    lenfr = math.ceil(len(ent)/(args.num+1))
    num = int(re.findall(r'-\d+\.h5', fopt, flags=0)[0][1:-3])
    if (num+1)*lenfr > len(ent):
        l = len(ent) - num*lenfr
    else:
        l = lenfr
    print('{} waveforms will be computed'.format(l))
    dt = np.zeros(l * (leng//5), dtype=opdt)
    start = 0
    end = 0
    for i in tqdm(range(num*lenfr, num*lenfr+l), disable=not Demo):
        cid = ent[i]['ChannelID']
        wave = wff.deduct_base(spe_pre[cid]['epulse'] * ent[i]['Waveform'], spe_pre[cid]['m_l'], spe_pre[cid]['thres'], 20, 'detail')
        pos = np.argwhere(wave[spe_pre[cid]['peak_c'] + 2:] > spe_pre[cid]['thres']).flatten()
        pf = wave[pos]/(spe_pre[cid]['spe'].sum())
        flag = 1
        if len(pos) == 0:
            flag = 0
        else:
            if not len(pos) in mcmc_collect:
                model_collect.update({len(pos) : partial(model, n=len(pos), eta=E)})
                nuts_kernel_collect.update({len(pos) : NUTS(model_collect[len(pos)], step_size=0.01, adapt_step_size=True)})
                mcmc_collect.update({len(pos) : MCMC(nuts_kernel_collect[len(pos)], num_warmup=warmup, num_samples=samples, num_chains=1, progress_bar=Demo, jit_model_args=True)})

            mne = spe[cid][np.mod(np.arange(leng).reshape(leng, 1) - pos.reshape(1, len(pos)), leng)]
            mcmc_collect[len(pos)].run(rng_key, wave=jnp.array(wave), mne=jnp.array(mne))
            # pf = np.mean(np.array(mcmc_collect[len(pos)].get_samples()['weight']), axis=0)
            pf = lasso_select(np.array(mcmc_collect[len(pos)].get_samples()['weight']), wave, mne)
            if len(pos) == 0:
                flag = 0
        if flag == 0:
            t = np.array([np.argmax(wave)]) - spe_pre[cid]['peak_c']
            pf = np.array([1])
            pos = t if t[0] >= 0 else np.array([0])
        pos_r = pos[pf > thres]
        pf = pf[pf > thres]
        lenpf = len(pf)
        end = start + lenpf
        dt['PETime'][start:end] = pos_r.astype(np.uint16)
        dt['Weight'][start:end] = pf.astype(np.float16)
        dt['EventID'][start:end] = ent[i]['EventID']
        dt['ChannelID'][start:end] = ent[i]['ChannelID']
        start = end
        if Demo :
            tth = ipt['GroundTruth']
            b = min(ent[i]['EventID']*30*len(Chnum), len(tth))
            tth = tth[0:b]
            j = np.where(np.logical_and(tth['EventID'] == ent[i]['EventID'], tth['ChannelID'] == cid))
            if len(tth[j]['PETime']) == 1:
                print('here')
            wff.demo(pos_r, pf, tth[j], spe_pre[cid], leng, pos, wave, cid)
    ipt.close()
    dt = dt[dt['Weight'] > 0]
    dt = np.sort(dt, kind='stable', order=['EventID', 'ChannelID', 'PETime'])
    with h5py.File(fopt, 'w') as opt:
        opt.create_dataset('Answer', data=dt, compression='gzip')
        print('The output file path is {}'.format(fopt), end=' ', flush=True)
    return

if __name__ == '__main__':
    main(args.opt, args.ipt, args.ref)
