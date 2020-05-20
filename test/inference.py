# -*- coding: utf-8 -*-

import sys
import re
import h5py
import math
import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
from jax import random
import numpyro
from numpyro.util import set_platform 
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import initialize_model
import argparse
import wf_func as wff

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', type=str, help='output file')
psr.add_argument('ipt', type=str, help='input file')
psr.add_argument('--met', type=str, help='fitting method')
psr.add_argument('--ref', type=str, help='reference file')
psr.add_argument('--num', type=int, help='fragment number')
psr.add_argument('--demo', dest='demo', action='store_true', help='demo bool', default=False)
args = psr.parse_args()

if args.demo:
    Demo = True

set_platform(platform='cpu')
N = 50

def lasso_select(pf_r, mne, wave):
    pf_r = pf_r[np.argmin([norm_fit_jnp(pf_r[j], mne, wave, eta=0) for j in range(len(pf_r))])]
    return pf_r

def norm_fit_jnp(x, M, y, eta=0):
    return np.power(y - np.matmul(M, x), 2).sum() + eta * x.sum()

def model(wave, mne):
    pf = numpyro.sample('weight', dist.HalfNormal(jnp.ones(N)))
    y = numpyro.sample('y', dist.Normal(0, 1), obs=wave-jnp.matmul(mne, pf))
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
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    nuts_kernel = NUTS(model, step_size=0.01, adapt_step_size=True)
    mcmc = MCMC(nuts_kernel, num_warmup=100, num_samples=500, num_chains=1, progress_bar=Demo, jit_model_args=True)
    for i in tqdm(range(num*lenfr, num*lenfr+l)):
        wave = wff.deduct_base(spe_pre[ent[i]['ChannelID']]['epulse'] * ent[i]['Waveform'], spe_pre[ent[i]['ChannelID']]['m_l'], spe_pre[ent[i]['ChannelID']]['thres'], 20, 'detail')
        possible = np.argpartition(wave[50:-50], -N)[-N:] + 50 - (spe_pre[ent[i]['ChannelID']]['peak_c'] + 2)
        flag = 1
        if len(possible) == 0:
            flag = 0
        else:
            mne = spe[ent[i]['ChannelID']][np.mod(np.arange(leng).reshape(leng, 1) - possible.reshape(1, len(possible)), leng)]
            mcmc.run(rng_key, wave=jnp.array(wave), mne=jnp.array(mne))
            pf = lasso_select(np.array(mcmc.get_samples()['weight']), mne, wave)
            pf = np.where(pf > 0, np.around(pf), 0)
            pos = possible[pf > 0]
            pf = pf[pf > 0]
            if len(possible) == 0:
                flag = 0
        if flag == 0:
            t = np.argwhere(wave == wave.min()).flatten()[0] - spe_pre[ent[i]['ChannelID']]['peak_c']
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
            j = np.where(np.logical_and(tth['EventID'] == ent[i]['EventID'], tth['ChannelID'] == ent[i]['ChannelID']))
            wff.demo(pos, pf, tth[j], spe_pre[ent[i]['ChannelID']], leng, possible, wave)
    ipt.close()
    dt = dt[dt['Weight'] > 0]
    dt = np.sort(dt, kind='stable', order=['EventID', 'ChannelID', 'PETime'])
    with h5py.File(fopt, 'w') as opt:
        opt.create_dataset('Answer', data=dt, compression='gzip')
        print('The output file path is {}'.format(fopt), end=' ', flush=True)
    return

if __name__ == '__main__':
    main(args.opt, args.ipt, args.ref)
