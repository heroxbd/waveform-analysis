# -*- coding: utf-8 -*-

import h5py
import numpy as np
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

psr = argparse.ArgumentParser()
psr.add_argument('ipt', type=str, help='input file')
psr.add_argument('--ref', type=str, help='reference file')

def model(wave, mne, n, eta):
    pf = numpyro.sample('weight', dist.HalfNormal(jnp.ones(n)))
    y = numpyro.sample('y', dist.Normal(0, 1), obs=jnp.power(wave-jnp.matmul(mne, pf), 2))
    return y

def norm_fit(x, M, y):
    return np.power(y - np.matmul(M, x), 2).sum()

def main(fopt, fipt, reference):
    spe_pre = wff.read_model(reference)
    ipt = h5py.File(fipt, 'r', libver='latest', swmr=True)
    ent = ipt['Waveform']
    Chnum = np.unique(ent['ChannelID'])
    spe = np.vstack([np.concatenate((spe_pre[i]['spe'], np.zeros(leng - len(spe_pre[i]['spe'])))) for i in Chnum])
    nuts_kernel = NUTS(model_collect[i], step_size=0.01, adapt_step_size=True)
    mcmc = MCMC(nuts_kernel_collect[i], num_warmup=50, num_samples=200, num_chains=1, progress_bar=True, jit_model_args=True)
    for i in range(len(ent)):
        cid = ent[i]['ChannelID']
        tth = ipt['GroundTruth']
        b = min(ent[i]['EventID']*30*len(Chnum), len(tth))
        tth = tth[0:b]
        j = np.where(np.logical_and(tth['EventID'] == ent[i]['EventID'], tth['ChannelID'] == cid))
        wave = spe_pre[cid]['epulse'] * ent[i]['Waveform']
        pos = np.argwhere(wave > spe_pre[cid]['epulse'])
        mne = spe[cid][np.mod(np.arange(leng).reshape(leng, 1) - pos.reshape(1, len(pos)), leng)]
        mcmc.run(rng_key, wave=jnp.array(wave), mne=jnp.array(mne))
        pf_samples = mcmc.get_samples()['weight']
        pf_mcmc = np.mean(pf_samples, axis=0)
        rss_mcmc = norm_fit(pf_mcmc, mne, wave)
        wff.demo(pos, pf_mcmc, tth[j], spe_pre[cid], leng, pos, wave)
        pf_l_bfgs_b = opti(wave, mne, len(pos))
        rss_mcmc = norm_fit(pf_l_bfgs_b, mne, wave)

    ipt.close()
    return

if __name__ == '__main__':
    main(args.opt, args.ipt, args.ref)
