# -*- coding: utf-8 -*-

import sys
import re
import h5py
import math
import numpy as np
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import initialize_model
import argparse
import wf_func as wff

Demo = False

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

def model(n, wave, mne, sigma):
    pf = numpyro.sample('weight', dist.HalfNormal(jnp.ones(n)))
    y = numpyro.sample('y', dist.Normal(0, sigma), obs=wave-jnp.matmul(mne, pf))
    return y

def main(fopt, fipt, reference):
    spe_pre = wff.read_model(reference)
    opdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint32), ('PETime', np.uint16), ('Weight', np.float16)])
    ipt = h5py.File(fipt, 'r', libver='latest', swmr=True)
    ent = ipt['Waveform']
    Chnum = np.unique(ent['ChannelID'])
    leng = len(ent[0]['Waveform'])
    assert leng >= len(spe_pre[1]['spe']), 'Single PE too long which is {}'.format(len(spe_pre[1]['spe']))
    spe = jnp.vstack([jnp.concatenate((jnp.array(spe_pre[i]['spe']), jnp.zeros(leng - len(spe_pre[i]['spe'])))) for i in Chnum])
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
    samp_list = [100, 500]; warm_list = [50, 100]
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    nuts_kernel = NUTS(model, step_size=0.01, adapt_step_size=True)
    mcmc0 = MCMC(nuts_kernel, num_warmup=warm_list[0], num_samples=samp_list[0])
    mcmc1 = MCMC(nuts_kernel, num_warmup=warm_list[1], num_samples=samp_list[1])
    for i in range(num*lenfr, num*lenfr+l):
        wave_i = wff.deduct_base(spe_pre[ent[i]['ChannelID']]['epulse'] * ent[i]['Waveform'], spe_pre[ent[i]['ChannelID']]['m_l'], spe_pre[ent[i]['ChannelID']]['thres'], 20, 'detail')
        pos = np.argwhere(wave_i > spe_pre[ent[i]['ChannelID']]['thres']).flatten() - (spe_pre[ent[i]['ChannelID']]['peak_c'] + 2)
        pos_r = jnp.array(pos[np.logical_and(pos >= 0, pos < leng)])
        wave = jnp.array(wave_i)
        flag = 1
        if len(pos) == 0:
            flag = 0
        else:
            mne = spe[ent[i]['ChannelID']][jnp.mod(jnp.arange(leng).reshape(leng, 1) - pos.reshape(1, len(pos)), leng)]
            mcmc0.run(rng_key, n=len(pos), wave=wave, mne=mne, sigma=0.5)
            pf_r = mcmc0.get_samples()['weight']
            pf_r = pf_r[jnp.array([norm_fit_jnp(pf_r[j], mne, wave, eta=jnp.sqrt(spe[ent[i]['ChannelID']].sum())/2) for j in range(samp_list[0])]).argmin()]
            pos_r = pos_r[pf_r > 0.05]
            if len(pos_r) == 0:
                flag = 0
            else:
                init_param = {'weight' : pf_r[pf_r > 0.05]}
                mne = spe[ent[i]['ChannelID']][jnp.mod(jnp.arange(leng).reshape(leng, 1) - pos_r.reshape(1, len(pos_r)), leng)]
                mcmc1.run(rng_key, n=len(pos_r), wave=wave, mne=mne, sigma=0.1)
                pf_r = mcmc1.get_samples()['weight']
                pf_r = pf_r[jnp.array([norm_fit_jnp(pf_r[j], mne, wave, eta=jnp.sqrt(spe[ent[i]['ChannelID']].sum())/2) for j in range(samp_list[1])]).argmin()]
                pf_r = jnp.around(pf_r)
                pf = np.array(pf_r[pf_r > 0.1])
                pos = np.array(pos_r[pf_r > 0.1])
                if len(pos) == 0:
                    flag = 0
        if flag == 0:
            t = np.argwhere(wave == wave.min()).flatten()[0] - spe_pre[ent[i]['ChannelID']]['peak_c']
            pf = np.array([1])
            pos = t if t[0] >= 0 else np.array([0])
        lenpf = len(pos)
        end = start + lenpf
        dt['PETime'][start:end] = pos.astype(np.uint16)
        dt['Weight'][start:end] = pf.astype(np.float16)
        dt['EventID'][start:end] = ent[i]['EventID']
        dt['ChannelID'][start:end] = ent[i]['ChannelID']
        start = end
        if Demo :
            pos_r = np.array(pos_r)
            tth = ipt['GroundTruth']
            b = min(ent[i]['EventID']*30*len(Chnum), len(tth))
            tth = tth[0:b]
            j = np.where(np.logical_and(tth['EventID'] == ent[i]['EventID'], tth['ChannelID'] == ent[i]['ChannelID']))
            wff.demo(pos, pf, tth[j], spe_pre[ent[i]['ChannelID']], leng, pos_r, wave_i)
    ipt.close()
    dt = dt[dt['Weight'] > 0]
    dt = np.sort(dt, kind='stable', order=['EventID', 'ChannelID', 'PETime'])
    with h5py.File(fopt, 'w') as opt:
        opt.create_dataset('Answer', data=dt, compression='gzip')
        print('The output file path is {}'.format(fopt), end=' ', flush=True)
    return

def norm_fit_jnp(x, M, y, eta=0):
    return jnp.power(y - jnp.matmul(M, x), 2).sum() + eta * x.sum()

if __name__ == '__main__':
    main(args.opt, args.ipt, args.ref)
