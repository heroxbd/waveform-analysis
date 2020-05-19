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

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', type=str, help='output file')
psr.add_argument('ipt', type=str, help='input file')
psr.add_argument('--met', type=str, help='fitting method')
psr.add_argument('--ref', type=str, help='reference file')
psr.add_argument('--num', type=int, help='fragment number')
args = psr.parse_args()

def model(n, wave, mne, sigma):
    pf = numpyro.sample('weight', dist.HalfNormal(jnp.ones(n)))
    y = numpyro.sample('y', dist.Normal(0, sigma), obs=wave-jnp.matmul(mne, pf))
    return y

def main(fopt, fipt, reference):
    samp_list = [100, 500]; warm_list = [50, 100]
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    nuts_kernel = NUTS(model, step_size=0.01, adapt_step_size=True)
    mcmc0 = MCMC(nuts_kernel, num_warmup=warm_list[0], num_samples=samp_list[0])
    mcmc1 = MCMC(nuts_kernel, num_warmup=warm_list[1], num_samples=samp_list[1])
    spe_pre = wff.read_model(reference)
    opdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint32), ('PETime', np.uint16), ('Weight', np.float16)])
    ipt = h5py.File(fipt, 'r', libver='latest', swmr=True)
    ent = ipt['Waveform']
    leng = len(ent[0]['Waveform'])
    assert leng >= len(spe_pre['spe']), 'Single PE too long which is {}'.format(len(spe_pre['spe']))
    spe = jnp.concatenate((jnp.array(spe_pre['spe']), jnp.zeros(leng - len(spe_pre['spe']))))
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
    for i in range(num*lenfr, num*lenfr+l):
        wave = wff.deduct_base(spe_pre['epulse'] * ent[i]['Waveform'], spe_pre['m_l'], spe_pre['thres'], 20, 'detail')
        pos = np.argwhere(wave > spe_pre['thres']).flatten() - (spe_pre['peak_c'] + 2)
        pos = jnp.array(pos[np.logical_and(pos >= 0, pos < l)])
        wave = jnp.array(wave)
        flag = 1
        if len(pos) == 0:
            flag = 0
        else:
            mne = spe[jnp.mod(jnp.arange(leng).reshape(leng, 1) - pos.reshape(1, len(pos)), leng)]
            mcmc0.run(rng_key, n=len(pos), wave=wave, mne=mne, sigma=0.5)
            pf_r = mcmc0.get_samples()['weight']
            pf_r = pf_r[jnp.array([norm_fit_jnp(pf_r[j], mne, wave, eta=0) for j in range(samp_list[0])]).argmin()]
            pos = pos[pf_r > 0.05]
            if len(pos) == 0:
                flag = 0
            else:
                init_param = {'weight' : pf_r[pf_r > 0.05]}
                mne = spe[jnp.mod(jnp.arange(leng).reshape(leng, 1) - pos.reshape(1, len(pos)), leng)]
                mcmc1.run(rng_key, n=len(pos), wave=wave, mne=mne, sigma=0.1)
                pf_r = mcmc1.get_samples()['weight']
                pf_r = pf_r[jnp.array([norm_fit_jnp(pf_r[j], mne, wave, eta=0) for j in range(samp_list[1])]).argmin()]
                pf_r = np.array(pf_r[pf_r > 0])
                pos_r = np.array(pos[pf_r > 0])
        if flag == 0:
            t = np.argwhere(wave == wave.min()).flatten()[0] - spe_pre['peak_c']
            pos_r = t if t[0] >= 0 else np.array([0])
            pf_r = np.array([1])
        pf = np.zeros(len(wave))
        pf[pos_r] = pf_r
        pet, pwe = wff.pf_to_tw(pf, 0.01)
        lenpf = len(pwe)
        end = start + lenpf
        dt['PETime'][start:end] = pet.astype(np.uint16)
        dt['Weight'][start:end] = pwe.astype(np.float16)
        dt['EventID'][start:end] = ent[i]['EventID']
        dt['ChannelID'][start:end] = ent[i]['ChannelID']
        start = end
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
