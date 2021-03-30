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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

spe_pre = wff.read_model(reference, 1)
with h5py.File(fipt, 'r', libver='latest', swmr=True) as ipt:
    ent = ipt['Readout/Waveform'][:]
    pelist = ipt['SimTriggerInfo/PEList'][:]
    t0_truth = ipt['SimTruth/T'][:]
    N = len(ent)
    print('{} waveforms will be computed'.format(N))
    window = len(ent[0]['Waveform'][::wff.nshannon])
    assert window >= len(spe_pre[0]['spe']), 'Single PE too long which is {}'.format(len(spe_pre[0]['spe']))
    Mu = ipt['Readout/Waveform'].attrs['mu'].item()
    Tau = ipt['Readout/Waveform'].attrs['tau'].item()
    Sigma = ipt['Readout/Waveform'].attrs['sigma'].item()
    gmu = ipt['SimTriggerInfo/PEList'].attrs['gmu'].item()
    gsigma = ipt['SimTriggerInfo/PEList'].attrs['gsigma'].item()
    s0 = spe_pre[0]['std'] / np.linalg.norm(spe_pre[0]['spe'])

p = spe_pre[0]['parameters']
if Tau != 0:
    Alpha = 1 / Tau
    Co = (Alpha / 2. * np.exp(Alpha ** 2 * Sigma ** 2 / 2.)).item()
std = 1.
Thres = {'mcmc':0.1, 'lucyddm':0.1, 'fbmp':1e-6}
mix0sigma = 1e-4

class mNormal(numpyro.distributions.distribution.Distribution):
    arg_constraints = {'pl': numpyro.distributions.constraints.real}
    support = numpyro.distributions.constraints.real
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
        return jax.scipy.special.logsumexp(prob, axis=0, b=pl)

def time_numpyro(a0, a1):
    nsp = 100
    nstd = -5
    Awindow = int(window * 0.95)
    rng_key = jax.random.PRNGKey(1)
    rng_key, rng_key_ = jax.random.split(rng_key)
    stime_t0 = np.empty(a1 - a0)
    stime_cha = np.empty(a1 - a0)
    accep = np.full(a1 - a0, np.nan)
    mix0ratio = np.full(a1 - a0, np.nan)
    dt = np.zeros((a1 - a0) * Awindow * 2, dtype=opdt)
    start = 0
    end = 0
    count = 0
    b = [0., 600.]
    time_mcmc = 0
    def model(n, y, mu, tlist, AV, t0left, t0right):
        t0 = numpyro.sample('t0', numpyro.distributions.Uniform(t0left, t0right))
        if Tau == 0:
            light_curve = numpyro.distributions.Normal(t0, scale=Sigma)
            pl = numpyro.primitives.deterministic('pl', jnp.exp(light_curve.log_prob(tlist)) / n * mu)
        else:
            pl = numpyro.primitives.deterministic('pl', Co * (1. - jax.scipy.special.erf((Alpha * Sigma ** 2 - (tlist - t0)) / (math.sqrt(2.) * Sigma))) * jnp.exp(-Alpha * (tlist - t0)) / n * mu)
        A = numpyro.sample('A', mNormal(pl, mix0sigma, 1., gsigma / gmu))
        with numpyro.plate('observations', len(y)):
            obs = numpyro.sample('obs', numpyro.distributions.Normal(jnp.matmul(AV, A), scale=std), obs=y)
        return obs
    for i in range(a0, a1):
        truth = pelist[pelist['TriggerNo'] == ent[i]['TriggerNo']]
        cid = ent[i]['ChannelID']
        wave = ent[i]['Waveform'].astype(np.float64) * spe_pre[cid]['epulse']

        n = max(math.ceil(Mu / math.sqrt(Tau ** 2 + Sigma ** 2)), 1)
        AV, wave, tlist, t0_init, t0_init_delta, A_init = wff.initial_params(wave, spe_pre[cid], Mu, Tau, Sigma, gmu, Thres['lucyddm'], p, nsp, nstd, is_t0=True, n=n, nshannon=1)
        AV = jnp.array(AV)
        wave = jnp.array(wave)
        tlist = jnp.array(tlist)
        t0_init = jnp.array(t0_init)
        A_init = jnp.array(A_init) + jnp.finfo(jnp.float64).epsneg

        time_mcmc_start = time.time()
        nuts_kernel = numpyro.infer.NUTS(model, adapt_step_size=True, init_strategy=numpyro.infer.initialization.init_to_value(values={'t0': t0_init, 'A': A_init}))
        mcmc = numpyro.infer.MCMC(nuts_kernel, num_samples=1000, num_warmup=1000, num_chains=1, progress_bar=False, chain_method='sequential', jit_model_args=True)
        try:
            ticrun = time.time()
            mcmc.run(rng_key, n=n, y=wave, mu=Mu, tlist=tlist, AV=AV, t0left=t0_init - 3 * Sigma, t0right=t0_init + 3 * Sigma, extra_fields=('num_steps', 'accept_prob', 'potential_energy'))
            tocrun = time.time()
            time_mcmc = time_mcmc + time.time() - time_mcmc_start
            num_leapfrogs = mcmc.get_extra_fields()['num_steps'].sum()
            # print('avg. time for each step :', (tocrun - ticrun) / num_leapfrogs)
            potential_energy = np.array(mcmc.get_extra_fields()['potential_energy'])
            accep[i - a0] = np.array(mcmc.get_extra_fields()['accept_prob']).mean()
            t0_t0 = np.array(mcmc.get_samples()['t0']).flatten()
            A = np.array(mcmc.get_samples()['A'])
            count = count + 1

            # # tlist_pan = np.array(tlist)
            tlist_pan = np.sort(np.unique(np.hstack(np.arange(0, window)[:, None] + np.arange(0, 1, 1 / n))))
            btlist = np.arange(t0_init - 3 * Sigma, t0_init + 3 * Sigma + 1e-6, 0.2)

            As = np.zeros(len(tlist_pan))
            As[np.isin(tlist_pan, tlist)] = np.mean(A, axis=0)
            logL = lambda t0 : -1 * np.sum(np.log(np.clip(wff.convolve_exp_norm(tlist_pan - t0, Tau, Sigma), np.finfo(np.float64).tiny, np.inf)) * As)
            logLv_btlist = np.vectorize(logL)(btlist)
            t0_cha = opti.fmin_l_bfgs_b(logL, x0=[btlist[np.argmin(logLv_btlist)]], approx_grad=True, bounds=[b], maxfun=50000)[0]
        except:
            time_mcmc = time_mcmc + time.time() - time_mcmc_start
            t0_t0 = np.array(t0_init)
            t0_cha = t0_init
            tlist = np.array(tlist)
            A = np.array([A_init])
            print('Failed waveform is TriggerNo = {:05d}, ChannelID = {:02d}, i = {:05d}'.format(ent[i]['TriggerNo'], cid, i))
        pet = np.array(tlist)
        cha = np.mean(A, axis=0)
        mix0ratio[i - a0] = (np.abs(cha) < 5 * mix0sigma).sum() / len(cha)
        pet, cha = wff.clip(pet, cha, 0)
        cha = cha * gmu
        t0_cha, _ = wff.likelihoodt0(pet, char=cha, gmu=gmu, Tau=Tau, Sigma=Sigma, mode='charge')
        stime_t0[i - a0] = np.mean(t0_t0)
        stime_cha[i - a0] = t0_cha
        end = start + len(cha)
        dt['HitPosInWindow'][start:end] = pet
        dt['Charge'][start:end] = cha
        dt['TriggerNo'][start:end] = ent[i]['TriggerNo']
        dt['ChannelID'][start:end] = ent[i]['ChannelID']
        start = end
    dt = dt[:end]
    dt = np.sort(dt, kind='stable', order=['TriggerNo', 'ChannelID'])
    return stime_t0, stime_cha, dt, count, time_mcmc, accep, mix0ratio

def fbmp_inference(a0, a1):
    nsp = 4
    nstd = 3
    D = 100
    stime_t0 = np.empty(a1 - a0)
    stime_cha = np.empty(a1 - a0)
    dt = np.zeros((a1 - a0) * window, dtype=opdt)
    d_tot = np.zeros(a1 - a0).astype(int)
    d_max = np.zeros(a1 - a0).astype(int)
    start = 0
    end = 0
    factor = np.linalg.norm(spe_pre[0]['spe'])
    b = [0., 600.]
    time_fbmp = 0
    for i in range(a0, a1):
        truth = pelist[pelist['TriggerNo'] == ent[i]['TriggerNo']]
        cid = ent[i]['ChannelID']
        wave = ent[i]['Waveform'].astype(np.float64) * spe_pre[cid]['epulse']

        n = max(math.ceil(Mu / math.sqrt(Tau ** 2 + Sigma ** 2)) * 1, 1)
        A, wave_r, tlist, t0_init, t0_init_delta, char_init = wff.initial_params(wave[::wff.nshannon], spe_pre[ent[i]['ChannelID']], Mu, Tau, Sigma, gmu, Thres['lucyddm'], p, nsp, nstd, is_t0=True, is_delta=False, n=n, nshannon=1)
        time_fbmp_start = time.time()
        A = np.matmul(A, np.diag(1. / np.sqrt(np.diag(np.matmul(A.T, A)))))
        xmmse, xmmse_star, psy_star, nu_star, T_star, d_tot_i, d_max_i = wff.fbmpr_fxn_reduced(wave_r, A, min(-1e-3+1, Mu / len(tlist)), spe_pre[cid]['std'] ** 2, (gsigma * factor / gmu) ** 2, factor, D, stop=0)
        # xmmse = np.zeros_like(tlist)
        # hitm = np.repeat(np.around(truth['HitPosInWindow'][:, None] * n) / n, len(tlist), axis=1)
        # xmmse = (np.repeat(truth['Charge'][:, None], len(tlist), axis=1) * (hitm == tlist)).sum(axis=0) * factor / gmu
        # xmmse_star = xmmse[None, :]
        # psy_star = np.array([1])
        # d_tot_i = 0
        # d_max_i = 0
        time_fbmp = time_fbmp + time.time() - time_fbmp_start

        tlist_pan = np.sort(np.unique(np.hstack(np.arange(0, window)[:, None] + np.arange(0, 1, 1 / n))))
        As = np.zeros((len(xmmse_star), len(tlist_pan)))
        As[:, np.isin(tlist_pan, tlist)] = np.clip(xmmse_star, 0, np.inf) / factor
        if sum(np.sum(As, axis=0) > 0):
            xmmse_most = xmmse_star[0]
            pet = tlist[xmmse_most > 0]
            cha = xmmse_most[xmmse_most > 0] / factor

            btlist = np.arange(t0_init - 3 * Sigma, t0_init + 3 * Sigma + 1e-6, 0.2)
            logL = lambda t0 : -1 * np.sum(special.logsumexp((np.log(np.clip(wff.convolve_exp_norm(tlist_pan - t0, Tau, Sigma), np.finfo(np.float64).tiny, np.inf))[None, :] * As).sum(axis=1), b=psy_star))
            logLv_btlist = np.vectorize(logL)(btlist)
            t0_t0 = opti.fmin_l_bfgs_b(logL, x0=[btlist[np.argmin(logLv_btlist)]], approx_grad=True, bounds=[b], maxfun=50000)[0]

            logL = lambda t0 : -1 * np.sum(np.log(np.clip(wff.convolve_exp_norm(tlist_pan - t0, Tau, Sigma), np.finfo(np.float64).tiny, np.inf)) * As[0][None, :])
            logLv_btlist = np.vectorize(logL)(btlist)
            t0_cha = opti.fmin_l_bfgs_b(logL, x0=[btlist[np.argmin(logLv_btlist)]], approx_grad=True, bounds=[b], maxfun=50000)[0]
        else:
            t0_t0 = t0_init
            t0_cha = t0_init
            d_tot_i = 0
            d_max_i = 0
            pet = tlist
            cha = char_init / gmu

        d_tot[i - a0] = d_tot_i
        d_max[i - a0] = d_max_i
        pet, cha = wff.clip(pet, cha, Thres[method])
        cha = cha * gmu
        stime_t0[i - a0] = t0_t0
        stime_cha[i - a0] = t0_cha
        end = start + len(cha)
        dt['HitPosInWindow'][start:end] = pet
        dt['Charge'][start:end] = cha
        dt['TriggerNo'][start:end] = ent[i]['TriggerNo']
        dt['ChannelID'][start:end] = ent[i]['ChannelID']
        start = end
    dt = dt[:end]
    dt = np.sort(dt, kind='stable', order=['TriggerNo', 'ChannelID'])
    return stime_t0, stime_cha, dt, d_tot, d_max, time_fbmp

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

opdt = np.dtype([('TriggerNo', np.uint32), ('ChannelID', np.uint32), ('HitPosInWindow', np.float64), ('Charge', np.float64)])

sdtp = np.dtype([('TriggerNo', np.uint32), ('ChannelID', np.uint32), ('tscharge', np.float64), ('tswave', np.float64)])
if method == 'mcmc':
    ts = np.zeros(N, dtype=sdtp)
    ts['TriggerNo'] = ent['TriggerNo']
    ts['ChannelID'] = ent['ChannelID']
    with Pool(min(args.Ncpu, cpu_count())) as pool:
        result = pool.starmap(partial(time_numpyro), slices)
    ts['tswave'] = np.hstack([result[i][0] for i in range(len(slices))])
    ts['tscharge'] = np.hstack([result[i][1] for i in range(len(slices))])
    As = np.hstack([result[i][2] for i in range(len(slices))])
    count = np.sum([result[i][3] for i in range(len(slices))])
    time_mcmc = np.hstack([result[i][4] for i in range(len(slices))]).mean()
    print('MCMC finished, real time {0:.02f}s'.format(time_mcmc))
    accep = np.hstack([result[i][5] for i in range(len(slices))])
    mix0ratio = np.hstack([result[i][6] for i in range(len(slices))])

    matplotlib.use('Agg')
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    ff = plt.figure(figsize=(16, 6))
    ax = ff.add_subplot(121)
    ax.hist(accep, bins=np.arange(0, 1+0.02, 0.02), label='accept_prob')
    ax.set_xlabel('accept_prob')
    ax.set_ylabel('Count')
    ax.legend(loc='upper left')
    ax = ff.add_subplot(122)
    ax.hist(mix0ratio, bins=np.arange(0, 1+0.02, 0.02), label='mix0ratio')
    ax.set_xlabel('mix0ratio')
    ax.set_ylabel('Count')
    ax.legend(loc='upper left')
    ff.savefig(os.path.splitext(fopt)[0] + '.png')
    plt.close()

    As = np.sort(As, kind='stable', order=['TriggerNo', 'ChannelID'])
    print('Successful MCMC ratio is {:.4%}'.format(count / N))
elif method == 'fbmp':
    ts = np.zeros(N, dtype=sdtp)
    ts['TriggerNo'] = ent['TriggerNo']
    ts['ChannelID'] = ent['ChannelID']
    with Pool(min(args.Ncpu, cpu_count())) as pool:
        result = pool.starmap(partial(fbmp_inference), slices)
    ts['tswave'] = np.hstack([result[i][0] for i in range(len(slices))])
    ts['tscharge'] = np.hstack([result[i][1] for i in range(len(slices))])
    As = np.hstack([result[i][2] for i in range(len(slices))])
    d_tot = np.hstack([result[i][3] for i in range(len(slices))])
    d_max = np.hstack([result[i][4] for i in range(len(slices))])
    time_fbmp = np.hstack([result[i][5] for i in range(len(slices))]).mean()
    print('FBMP finished, real time {0:.02f}s'.format(time_fbmp))

    matplotlib.use('Agg')
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    ff = plt.figure(figsize=(16, 6))
    ax = ff.add_subplot(121)
    # di, ci = np.unique(d_tot, return_counts=True)
    # ax.bar(di, ci, label='d_tot')
    ax.hist(d_tot, bins=np.linspace(0, d_tot.max(), 20), label='d_tot')
    ax.set_xlabel('d_tot')
    ax.set_ylabel('Count')
    ax.set_ylim(0.5, N)
    ax.set_yscale('log')
    ax.legend(loc='upper center')
    ax = ff.add_subplot(122)
    # di, ci = np.unique(d_max, return_counts=True)
    # ax.bar(di, ci, label='d_max')
    ax.hist(d_tot, bins=np.linspace(0, d_tot.max(), 20), label='d_max')
    ax.set_xlabel('d_max')
    ax.set_ylabel('Count')
    ax.set_ylim(0.5, N)
    ax.set_yscale('log')
    ax.legend(loc='upper right')
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