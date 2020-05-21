# -*- coding: utf-8 -*-

import sys
import re
import h5py
import math
import numpy as np
from tqdm import tqdm
from scipy import optimize as opti
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
thres = 0.1
N = [50, 100]
warmup = [50, 100]
samples = [50, 200]
M = [14, 50]
E = 2000

def lasso_select(pf_r, wave, mne):
    pf_r = pf_r[np.argmin([norm_fit(pf_r[j], mne, wave, eta=0) for j in range(len(pf_r))])]
    return pf_r

def norm_fit(x, M, y, eta=0):
    return np.power(y - np.matmul(M, x), 2).sum() + eta * x.sum()

def model_light(wave, mne):
    pf = numpyro.sample('weight', dist.HalfNormal(jnp.ones(N[0])))
    y = numpyro.sample('y', dist.Normal(0, 1), obs=wave-jnp.matmul(mne, pf))
    return y

def model_heavy(wave, mne):
    pf = numpyro.sample('weight', dist.HalfNormal(jnp.ones(N[1])))
    y = numpyro.sample('y', dist.Normal(0, 1), obs=wave-jnp.matmul(mne, pf))
    return y

def polyopti(wave, mne, n, it=50000):
    ans = np.ones(n).astype(np.float64) * 0.3
    b = np.zeros((n, 2)).astype(np.float64); b[:, 1] = np.inf
    ans = opti.fmin_l_bfgs_b(norm_fit, ans, args=(mne, wave, E), approx_grad=True, bounds=b, maxfun=it)
    return ans[0]

def ergodic(wave, mne, base, n):
    omega = 2**n
    b = np.argmin([norm_fit(np.array(list(['{:0'+str(n)+'b}'][0].format(i))).astype(np.float16) + base, mne, wave, eta=0) for i in range(omega)])
    return np.array(list(['{:0'+str(n)+'b}'][0].format(b))).astype(np.float16) + base

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
    nuts_kernel_light = NUTS(model_light, step_size=0.01, adapt_step_size=True)
    nuts_kernel_heavy = NUTS(model_heavy, step_size=0.01, adapt_step_size=True)
    mcmc_light = MCMC(nuts_kernel_light, num_warmup=warmup[0], num_samples=samples[0], num_chains=1, progress_bar=Demo, jit_model_args=True)
    mcmc_heavy = MCMC(nuts_kernel_heavy, num_warmup=warmup[1], num_samples=samples[1], num_chains=1, progress_bar=Demo, jit_model_args=True)

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
        pos = np.argwhere(wave[spe_pre[cid]['peak_c']:] > spe_pre[cid]['thres']).flatten() - 2
        flag = 1
        if len(pos) == 0:
            flag = 0
        elif len(pos) > 0 and len(pos) <= M[1]:
            mne = spe[cid][np.mod(np.arange(leng).reshape(leng, 1) - pos.reshape(1, len(pos)), leng)]
            pf = polyopti(wave, mne, len(pos))
        else:
            pos = np.sort(np.argpartition(wave[spe_pre[cid]['peak_c']:], -N[1])[-N[1]:]) - 2
            mne = spe[cid][np.mod(np.arange(leng).reshape(leng, 1) - pos.reshape(1, len(pos)), leng)]
            mcmc_heavy.run(rng_key, wave=jnp.array(wave), mne=jnp.array(mne))
            pf = lasso_select(np.array(mcmc_heavy.get_samples()['weight']), wave, mne)
        pos = pos[pf > thres]
        pf = pf[pf > thres]
        if len(pos) == 0:
            flag = 0
        elif len(pos) > M[0]:
            if len(pos) <= M[1]:
                mne = spe[cid][np.mod(np.arange(leng).reshape(leng, 1) - pos.reshape(1, len(pos)), leng)]
                pf = polyopti(wave, mne, len(pos))
            else: 
                pos = np.sort(pos[np.argpartition(pf, -N[0])[-N[0]:]])
                mne = spe[cid][np.mod(np.arange(leng).reshape(leng, 1) - pos.reshape(1, len(pos)), leng)]
                mcmc_light.run(rng_key, wave=jnp.array(wave), mne=jnp.array(mne))
                pf = lasso_select(np.array(mcmc_light.get_samples()['weight']), wave, mne)
            pos = pos[pf > thres]
            pf = pf[pf > thres]
            if len(pos) == 0:
                flag = 0
            elif len(pos) > M[0]:
                if len(pos) <= M[1]:
                    mne = spe[cid][np.mod(np.arange(leng).reshape(leng, 1) - pos.reshape(1, len(pos)), leng)]
                    pf = polyopti(wave, mne, len(pos), it=500000)
                else: 
                    pos = np.sort(pos[np.argpartition(pf, -N[0])[-N[0]:]])
                    mne = spe[cid][np.mod(np.arange(leng).reshape(leng, 1) - pos.reshape(1, len(pos)), leng)]
                    mcmc_light.run(rng_key, wave=jnp.array(wave), mne=jnp.array(mne))
                    pf = lasso_select(np.array(mcmc_light.get_samples()['weight']), wave, mne)
            else:
                mne = spe[cid][np.mod(np.arange(leng).reshape(leng, 1) - pos.reshape(1, len(pos)), leng)]
                base = np.floor(pf)
                pf = ergodic(wave, mne, base, len(pos))
        else:
            mne = spe[cid][np.mod(np.arange(leng).reshape(leng, 1) - pos.reshape(1, len(pos)), leng)]
            base = np.floor(pf)
            pf = ergodic(wave, mne, base, len(pos))
        pf = np.around(pf)
        pos = pos[pf > thres]
        pf = pf[pf > thres]
        if flag == 0:
            t = np.argwhere(wave == wave.min()).flatten()[:1] - spe_pre[cid]['peak_c']
            pf = np.array([1])
            pos = t if t[0] >= 0 else np.array([0])
        lenpf = len(pf)
        end = start + lenpf
        dt['PETime'][start:end] = pos.astype(np.uint16)
        dt['Weight'][start:end] = pf.astype(np.float16)
        dt['EventID'][start:end] = ent[i]['EventID']
        dt['ChannelID'][start:end] = ent[i]['ChannelID']
        start = end
        if Demo :
            tth = ipt['GroundTruth']
            b = min(ent[i]['EventID']*30*len(Chnum), len(tth))
            tth = tth[0:b]
            j = np.where(np.logical_and(tth['EventID'] == ent[i]['EventID'], tth['ChannelID'] == cid))
            wff.demo(pos, pf, tth[j], spe_pre[cid], leng, pos, wave)
    ipt.close()
    dt = dt[dt['Weight'] > 0]
    dt = np.sort(dt, kind='stable', order=['EventID', 'ChannelID', 'PETime'])
    with h5py.File(fopt, 'w') as opt:
        opt.create_dataset('Answer', data=dt, compression='gzip')
        print('The output file path is {}'.format(fopt), end=' ', flush=True)
    return

if __name__ == '__main__':
    main(args.opt, args.ipt, args.ref)
