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
import scipy.stats
from scipy.stats import poisson, uniform, norm
import scipy.special as special
from scipy import optimize as opti
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
    pelist = ipt['SimTriggerInfo/PEList'][:]
    t0_truth = ipt['SimTruth/T'][:]
    N = len(ent)
    print('{} waveforms will be computed'.format(N))
    window = len(ent[0]['Waveform'])
    assert window >= len(spe_pre[0]['spe']), 'Single PE too long which is {}'.format(len(spe_pre[0]['spe']))
    Mu = ipt['Readout/Waveform'].attrs['mu'].item()
    Tau = ipt['Readout/Waveform'].attrs['tau'].item()
    Sigma = ipt['Readout/Waveform'].attrs['sigma'].item()
    gmu = ipt['SimTriggerInfo/PEList'].attrs['gmu'].item()
    gsigma = ipt['SimTriggerInfo/PEList'].attrs['gsigma'].item()
    npe = ipt['SimTruth/T'].attrs['npe'].item()

p = spe_pre[0]['parameters']
if Tau != 0:
    Alpha = 1 / Tau
    Co = (Alpha / 2. * np.exp(Alpha ** 2 * Sigma ** 2 / 2.)).item()
std = 1.
Thres = {'mcmc':0.2, 'lucyddm':0.2, 'fbmp':0.2}

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
    nsp = 4
    Awindow = int(window * 0.95)
    rng_key = jax.random.PRNGKey(1)
    rng_key, rng_key_ = jax.random.split(rng_key)
    stime = np.empty(a1 - a0)
    accep = np.full(a1 - a0, np.nan)
    dt = np.zeros((a1 - a0) * Awindow * 2, dtype=opdt)
    start = 0
    end = 0
    count = 0
    def model(n, y, mu, tlist, AV, t0left, t0right):
        t0 = numpyro.sample('t0', numpyro.distributions.Uniform(t0left, t0right))
        if Tau == 0:
            light_curve = numpyro.distributions.Normal(t0, scale=Sigma)
            pl = numpyro.primitives.deterministic('pl', jnp.exp(light_curve.log_prob(tlist)) / n * mu + jnp.finfo(jnp.float64).epsneg)
        else:
            pl = numpyro.primitives.deterministic('pl', Co * (1. - jax.scipy.special.erf((Alpha * Sigma ** 2 - (tlist - t0)) / (math.sqrt(2.) * Sigma))) * jnp.exp(-Alpha * (tlist - t0)) / n * mu + jnp.finfo(jnp.float64).epsneg)
        A = numpyro.sample('A', mNormal(pl, std / gsigma, 1., gsigma / gmu))
        with numpyro.plate('observations', len(y)):
            obs = numpyro.sample('obs', numpyro.distributions.Normal(jnp.matmul(AV, A), scale=std), obs=y)
        return obs
    for i in range(a0, a1):
        truth = pelist[pelist['TriggerNo'] == ent[i]['TriggerNo']]
        cid = ent[i]['ChannelID']
        wave = ent[i]['Waveform'].astype(np.float64) * spe_pre[cid]['epulse']

        AV, wave, tlist, t0_init, t0_init_delta, A_init, mu, n = wff.initial_params(wave, spe_pre[cid], Tau, Sigma, gmu, gsigma, Thres['lucyddm'], npe, p, nsp, is_t0=True)
        AV = jnp.array(AV)
        wave = jnp.array(wave)
        tlist = jnp.array(tlist)
        t0_init = jnp.array(t0_init)
        A_init = jnp.array(A_init) + jnp.finfo(jnp.float64).epsneg
        mu = jnp.array(mu)

        nuts_kernel = numpyro.infer.NUTS(model, adapt_step_size=True, init_strategy=numpyro.infer.initialization.init_to_value(values={'t0': t0_init, 'A': A_init}))
        mcmc = numpyro.infer.MCMC(nuts_kernel, num_samples=1000, num_warmup=500, num_chains=1, progress_bar=Demo, chain_method='sequential', jit_model_args=True)
        try:
            ticrun = time.time()
            mcmc.run(rng_key, n=n, y=wave, mu=mu, tlist=tlist, AV=AV, t0left=t0_init - 3 * Sigma, t0right=t0_init + 3 * Sigma, extra_fields=('num_steps', 'accept_prob'))
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
            t0 = np.array(t0_init)
            tlist = np.array(tlist)
            A = np.array(A_init)
            print('Failed waveform is TriggerNo = {:05d}, ChannelID = {:02d}, i = {:05d}'.format(ent[i]['TriggerNo'], ent[i]['ChannelID'], i))
        stime[i - a0] = np.mean(t0)
        pet, cha = wff.clip(np.array(tlist), np.mean(A, axis=0), Thres['mcmc'])
        cha = cha / cha.sum() * np.clip(np.abs(wave.sum()), 1e-6, np.inf)
        end = start + len(cha)
        dt['HitPosInWindow'][start:end] = pet
        dt['Charge'][start:end] = cha
        dt['TriggerNo'][start:end] = ent[i]['TriggerNo']
        dt['ChannelID'][start:end] = ent[i]['ChannelID']
        start = end
    dt = dt[:end]
    dt = np.sort(dt, kind='stable', order=['TriggerNo', 'ChannelID'])
    return stime, dt, count, accep

def fbmp_inference(a0, a1):
    nsp = 4
    D = 10
    stime = np.empty(a1 - a0)
    dt = np.zeros((a1 - a0) * window, dtype=opdt)
    d_tot = np.zeros(a1 - a0).astype(int)
    start = 0
    end = 0
    factor = np.linalg.norm(spe_pre[0]['spe'])
    tlist = np.arange(0, window - len(spe_pre[0]['spe']))
    t_auto = np.arange(window)[:, None] - tlist
    A = p[2] * np.exp(-1 / 2 * np.power((np.log((t_auto + np.abs(t_auto)) * (1 / p[0] / 2)) * (1 / p[1])), 2))
    A = A / factor
    # A = np.matmul(A, np.diag(1. / np.sqrt(np.diag(np.matmul(A.T, A)))))
    b = [0., 600.]
    time_fbmp = 0
    def loglikelihood(t0, tlist, xmmse, psy_star):
        if Tau == 0:
            pl = 1 / (math.sqrt(2. * math.pi) * Sigma) * np.exp(-((tlist - t0) / Sigma) ** 2 / 2)
        else:
            pl = Co * (1. - special.erf((Alpha * Sigma ** 2 - (tlist - t0)) / (math.sqrt(2.) * Sigma))) * np.exp(-Alpha * (tlist - t0))
        logL = special.logsumexp(np.sum(special.logsumexp(np.einsum('ijk->jik', np.stack([norm.logpdf(xmmse, loc=0, scale=std), norm.logpdf(xmmse, loc=gmu, scale=gsigma)])), axis=1, b=np.stack([(1 - pl), pl])), axis=1), axis=0, b=psy_star)
        return logL
    for i in range(a0, a1):
        truth = pelist[pelist['TriggerNo'] == ent[i]['TriggerNo']]
        cid = ent[i]['ChannelID']
        wave = ent[i]['Waveform'].astype(np.float64) * spe_pre[cid]['epulse']

        time_fbmp_start = time.time()
        A, wave_r, tlist, t0_init, t0_init_delta, char_init, mu, n = wff.initial_params(wave, spe_pre[ent[i]['ChannelID']], Tau, Sigma, gmu, gsigma, Thres['lucyddm'], npe, p, nsp, is_t0=True)
        A = A / factor
        # A = np.matmul(A, np.diag(1. / np.sqrt(np.diag(np.matmul(A.T, A)))))
        xmmse, xmmse_star_r, psy_star, nu_star, T_star, d_tot_i, d_max = wff.fbmpr_fxn_reduced(wave_r, A, mu / len(tlist), spe_pre[cid]['std'] ** 2, gsigma ** 2 * factor / gmu, factor, D, stop=0)
        time_fbmp = time_fbmp + time.time() - time_fbmp_start

        xmmse_star = np.clip(xmmse_star_r, 0, np.inf)
        xmmse_star_vali = xmmse_star[:, np.sum(xmmse_star, axis=0) > 0] / factor * gmu
        tlist_vali = tlist[np.sum(xmmse_star, axis=0) > 0]
        logL = lambda t0 : -1 * loglikelihood(t0, tlist_vali, xmmse_star_vali, psy_star)
        logLv = np.vectorize(logL)
        btlist = np.arange(t0_init - 3 * Sigma, t0_init + 3 * Sigma + 1e-6, 0.2)
        t0 = opti.fmin_l_bfgs_b(logL, x0=[btlist[np.argmin(logLv(btlist))]], approx_grad=True, bounds=[b], maxfun=50000)[0]
        # logLvdelta = np.vectorize(lambda t : np.abs(logL(t) - logL(t0) - 0.5))
        # t0delta = abs(opti.fmin_l_bfgs_b(logLvdelta, x0=[btlist[np.argmin(logLvdelta(btlist))]], approx_grad=True, bounds=[b], maxfun=50000)[0] - t0)

        stime[i - a0] = t0
        d_tot[i - a0] = d_tot_i
        xmmse_most = xmmse_star_r[0]
        pet = tlist[xmmse_most > 0]
        cha = xmmse_most[xmmse_most > 0] / factor
        pet, cha = wff.clip(pet, cha, Thres['fbmp'])
        cha = cha / cha.sum() * np.clip(np.abs(wave.sum()), 1e-6, np.inf)
        end = start + len(cha)
        dt['HitPosInWindow'][start:end] = pet
        dt['Charge'][start:end] = cha
        dt['TriggerNo'][start:end] = ent[i]['TriggerNo']
        dt['ChannelID'][start:end] = ent[i]['ChannelID']
        start = end
    dt = dt[:end]
    dt = np.sort(dt, kind='stable', order=['TriggerNo', 'ChannelID'])
    return stime, dt, d_tot, time_fbmp

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
opdt = np.dtype([('TriggerNo', np.uint32), ('ChannelID', np.uint32), ('HitPosInWindow', np.float64), ('Charge', np.float64)])

if method == 'mcmc':
    with Pool(min(args.Ncpu, cpu_count())) as pool:
        result = pool.starmap(partial(time_numpyro), slices)
    ts['tswave'] = np.hstack([result[i][0] for i in range(len(slices))])
    As = np.hstack([result[i][1] for i in range(len(slices))])
    count = np.sum([result[i][2] for i in range(len(slices))])
    accep = np.hstack([result[i][3] for i in range(len(slices))])

    ff = plt.figure(figsize=(8, 6))
    ax = ff.add_subplot()
    ax.hist(accep, bins=np.arange(0, 1+0.02, 0.02), label='accept_prob')
    ax.legend(loc='upper left')
    ff.savefig(os.path.splitext(fopt)[0] + '.png')
    plt.close()

    As = np.sort(As, kind='stable', order=['TriggerNo', 'ChannelID'])
    print('Successful MCMC ratio is {:.4%}'.format(count / N))
elif method == 'fbmp':
    with Pool(min(args.Ncpu, cpu_count())) as pool:
        result = pool.starmap(partial(fbmp_inference), slices)
    ts['tswave'] = np.hstack([result[i][0] for i in range(len(slices))])
    As = np.hstack([result[i][1] for i in range(len(slices))])
    d_tot = np.hstack([result[i][2] for i in range(len(slices))])
    time_fbmp = np.sum(np.hstack([result[i][3] for i in range(len(slices))])) / args.Ncpu
    print('FBMP finished, real time {0:.02f}s per core'.format(time_fbmp))

    ff = plt.figure(figsize=(8, 6))
    ax = ff.add_subplot()
    di, ci = np.unique(d_tot, return_counts=True)
    ax.bar(di, ci, label='d_tot')
    ax.legend()
    ff.savefig(os.path.splitext(fopt)[0] + '.png')
    plt.close()
    As = np.sort(As, kind='stable', order=['TriggerNo', 'ChannelID'])

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