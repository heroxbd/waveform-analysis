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
import scipy.integrate as integrate
from scipy import optimize as opti
import scipy.special as special
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
    pan = np.arange(window)
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
Thres = {'mcmc':std / gsigma, 'lucyddm':0.1, 'fbmp':1e-6}
# Thres = {'mcmc':std / gsigma, 'lucyddm':0.01, 'fbmp':1e-6}
mix0sigma = 1e-3
mu0 = np.arange(1, int(Mu + 5 * np.sqrt(Mu)))
n_t = np.arange(1, 20)
p_t = special.comb(mu0, 2)[:, None] * np.power(wff.convolve_exp_norm(np.arange(1029) - 200, Tau, Sigma) / n_t[:, None], 2).sum(axis=1)
n0 = np.array([n_t[p_t[i] < max(1e-1, np.sort(p_t[i])[1])].min() for i in range(len(mu0))])
ndict = dict(zip(mu0, n0))

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
    nsp = 4
    nstd = 3
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

        mu = abs(wave.sum() / gmu)
        n = ndict[min(math.ceil(mu), max(mu0))]
        AV, wave, tlist, t0_init, t0_init_delta, A_init, left_wave, right_wave = wff.initial_params(wave[::wff.nshannon], spe_pre[cid], Tau, Sigma, gmu, Thres['lucyddm'], p, nsp, nstd, is_t0=True, n=n, nshannon=1)
        mu = abs(wave.sum() / gmu)
        AV = jnp.array(AV)
        wave = jnp.array(wave)
        tlist = jnp.array(tlist)
        t0_init = jnp.array(t0_init)
        A_init = jnp.array(A_init)

        time_mcmc_start = time.time()
        nuts_kernel = numpyro.infer.NUTS(model, adapt_step_size=True, init_strategy=numpyro.infer.initialization.init_to_value(values={'t0': t0_init, 'A': A_init}))
        mcmc = numpyro.infer.MCMC(nuts_kernel, num_samples=1000, num_warmup=1000, num_chains=1, progress_bar=False, chain_method='sequential', jit_model_args=True)
        try:
            ticrun = time.time()
            mcmc.run(rng_key, n=n, y=wave, mu=mu, tlist=tlist, AV=AV, t0left=t0_init - 3 * Sigma, t0right=t0_init + 3 * Sigma, extra_fields=('accept_prob', 'potential_energy'))
            tocrun = time.time()
            time_mcmc = time_mcmc + time.time() - time_mcmc_start
            potential_energy = np.array(mcmc.get_extra_fields()['potential_energy'])
            accep[i - a0] = np.array(mcmc.get_extra_fields()['accept_prob']).mean()
            t0_t0 = np.array(mcmc.get_samples()['t0']).flatten()
            A = np.array(mcmc.get_samples()['A'])
            count = count + 1
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
    prior = True
    space = True
    elbo = True
    nsp = 4
    nstd = 3
    t0_wav = np.empty(a1 - a0)
    t0_cha = np.empty(a1 - a0)
    mu_wav = np.empty(a1 - a0)
    mu_cha = np.empty(a1 - a0)
    dt = np.zeros((a1 - a0) * window, dtype=opdt)
    d_tot = np.zeros(a1 - a0).astype(int)
    d_max = np.zeros(a1 - a0).astype(int)
    nu_truth = np.empty(a1 - a0)
    nu_max = np.empty(a1 - a0)
    num = np.zeros(a1 - a0).astype(int)
    start = 0
    end = 0
    b_t0 = [0., 600.]
    time_fbmp = 0
    for i in range(a0, a1):
        cid = ent[i]['ChannelID']
        wave = ent[i]['Waveform'].astype(np.float64) * spe_pre[cid]['epulse']

        # initialization
        mu_t = abs(wave.sum() / gmu)
        n = min(max(round(20 / mu_t), 1), 10)
        A, wave_r, tlist, t0_t, t0_delta, cha, left_wave, right_wave = wff.initial_params(wave[::wff.nshannon], spe_pre[ent[i]['ChannelID']], Tau, Sigma, gmu, Thres['lucyddm'], p, nsp, nstd, is_t0=True, is_delta=False, n=n, nshannon=1)
        mu_t = abs(wave_r.sum() / gmu)
        def optit0mu(t0, mu, n, xmmse_star, psy_star, c_star, la):
            if prior:
                ys = np.log(psy_star) - np.log(poisson.pmf(c_star, la)).sum(axis=1)
            ys = np.exp(ys - ys.max()) / np.sum(np.exp(ys - ys.max()))
            t0list = np.arange(t0 - 3 * Sigma, t0 + 3 * Sigma + 1e-6, 0.2)
            mulist = np.arange(max(1e-8, mu - 3 * np.sqrt(mu)), mu + 3 * np.sqrt(mu), 0.1)
            b_mu = [max(1e-8, mu - 5 * np.sqrt(mu)), mu + 5 * np.sqrt(mu)]
            tlist_pan = np.sort(np.unique(np.hstack(np.arange(0, window)[:, None] + np.arange(0, 1, 1 / n))))
            As = np.zeros((len(xmmse_star), len(tlist_pan)))
            As[:, np.isin(tlist_pan, tlist)] = c_star
            assert sum(np.sum(As, axis=0) > 0) > 0
            def likelihood(x):
                a = x[0] * wff.convolve_exp_norm(tlist_pan - x[1], Tau, Sigma) / n + 1e-8 # use tlist_pan not tlist
                li = -special.logsumexp(np.log(poisson.pmf(As, a)).sum(axis=1), b=ys)
                return li
            likemu = np.array([likelihood([mulist[j], t0]) for j in range(len(mulist))])
            liket0 = np.array([likelihood([mu, t0list[j]]) for j in range(len(t0list))])
            mu, t0 = opti.fmin_l_bfgs_b(likelihood, x0=[mulist[likemu.argmin()], t0list[liket0.argmin()]], approx_grad=True, bounds=[b_mu, b_t0], maxfun=50000)[0]
            return mu, t0

        truth = pelist[pelist['TriggerNo'] == ent[i]['TriggerNo']]

        # tlist = truth['HitPosInWindow'][truth['HitPosInWindow'] < right_wave - 1]
        # t_auto = (np.arange(left_wave, right_wave) / wff.nshannon)[:, None] - tlist
        # A = wff.spe((t_auto + np.abs(t_auto)) / 2, p[0], p[1], p[2])

        # t0_t = t0_truth[i]['T0']
        # mu_t = len(truth)
        # 1st FBMP
        time_fbmp_start = time.time()
        # Eq. (9) where the columns of A are taken to be unit-norm.
        factor = np.sqrt(np.diag(np.matmul(A.T, A)))
        A = A / factor
        # la = mu_t * wff.convolve_exp_norm(tlist - t0_t, Tau, Sigma) / n + 1e-8
        # la = mu_t * np.ones(len(tlist)) / len(tlist)
        la = np.ones(len(tlist))
        # la = mu_t * np.ones(len(tlist))
        # la = cha / cha.sum() * mu_t + 1e-8
        # la = la / la.sum() * mu_t
        xmmse, xmmse_star, psy_star, nu_star, nu_star_bk, T_star, d_max_i, num_i = wff.fbmpr_fxn_reduced(wave_r, A, la, spe_pre[cid]['std'] ** 2, (gsigma * factor / gmu) ** 2, factor, len(la), stop=5, truth=truth, i=i, left=left_wave, right=right_wave, tlist=tlist, gmu=gmu, para=p, prior=prior, space=space, elbo=elbo)
        time_fbmp = time_fbmp + time.time() - time_fbmp_start
        c_star = np.zeros_like(xmmse_star).astype(int)
        for k in range(len(T_star)):
            t, c = np.unique(T_star[k][xmmse_star[k][T_star[k]] > 0], return_counts=True)
            c_star[k, t] = c
        maxindex = psy_star.argmax()

        nx = np.sum([(tlist - 0.5 / n <= truth['HitPosInWindow'][j]) * (tlist + 0.5 / n > truth['HitPosInWindow'][j]) for j in range(len(truth))], axis=0)
        cc = np.sum([np.where(tlist - 0.5 / n < truth['HitPosInWindow'][j], np.sqrt(truth['Charge'][j]), 0) * np.where(tlist + 0.5 / n > truth['HitPosInWindow'][j], np.sqrt(truth['Charge'][j]), 0) for j in range(len(truth))], axis=0)
        c_star_truth = np.sum([np.where(tlist - 0.5 / n < truth['HitPosInWindow'][j], 1, 0) * np.where(tlist + 0.5 / n > truth['HitPosInWindow'][j], 1, 0) for j in range(len(truth))], axis=0)
        if c_star_truth.sum() == 0:
            c_star_truth[0] = 1
        pp = np.arange(left_wave, right_wave)
        wav_ans = np.sum([np.where(pp > truth['HitPosInWindow'][j], wff.spe(pp - truth['HitPosInWindow'][j], tau=p[0], sigma=p[1], A=p[2]) * truth['Charge'][j] / gmu, 0) for j in range(len(truth))], axis=0)
        nu_truth[i - a0] = wff.nu_direct(wave_r, A, nx, factor, (gsigma * factor / gmu) ** 2, spe_pre[cid]['std'] ** 2, la, prior=prior, space=space)
        rss_truth = np.power(wav_ans - np.matmul(A, cc / gmu * factor), 2).sum()
        nu = np.array([wff.nu_direct(wave_r, A, c_star[j], factor, (gsigma * factor / gmu) ** 2, spe_pre[cid]['std'] ** 2, la, prior=prior, space=space) for j in range(len(psy_star))])
        nu_max[i - a0] = nu[0]
        rss = np.array([np.power(wav_ans - np.matmul(A, xmmse_star[j]), 2).sum() for j in range(len(psy_star))])

        xmmse_most = xmmse_star[maxindex]
        pet = np.repeat(tlist[xmmse_most > 0], c_star[maxindex][xmmse_most > 0])
        cha = np.repeat(xmmse_most[xmmse_most > 0] / factor[xmmse_most > 0] / c_star[maxindex][xmmse_most > 0], c_star[maxindex][xmmse_most > 0])

        mu = np.average(c_star.sum(axis=1), weights=psy_star)
        t0 = t0_t
        mu_i = len(cha)
        t0_i = t0_t

        # mu, t0 = optit0mu(t0_t, mu_t, n, np.empty(len(la))[None, :], np.array([1]), c_star_truth[None, :], la)

        # mu, t0 = optit0mu(t0_t, mu_t, n, xmmse_star, psy_star, c_star, la)
        # mu_i, t0_i = optit0mu(t0_t, mu_t, n, xmmse_most[None, :], np.array([1]), c_star[maxindex][None, :], la)

        d_tot[i - a0] = len(la)
        d_max[i - a0] = d_max_i
        pet, cha = wff.clip(pet, cha, Thres[method])
        cha = cha * gmu
        t0_wav[i - a0] = t0
        t0_cha[i - a0] = t0_i
        mu_wav[i - a0] = mu
        mu_cha[i - a0] = mu_i
        num[i - a0] = num_i
        end = start + len(cha)
        dt['HitPosInWindow'][start:end] = pet
        dt['Charge'][start:end] = cha
        dt['TriggerNo'][start:end] = ent[i]['TriggerNo']
        dt['ChannelID'][start:end] = ent[i]['ChannelID']
        start = end
    dt = dt[:end]
    dt = np.sort(dt, kind='stable', order=['TriggerNo', 'ChannelID'])
    return t0_wav, t0_cha, dt, mu_wav, mu_cha, d_tot, d_max, time_fbmp, nu_truth, nu_max, num

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

if method == 'mcmc':
    sdtp = np.dtype([('TriggerNo', np.uint32), ('ChannelID', np.uint32), ('tscharge', np.float64), ('tswave', np.float64)])
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
    sdtp = np.dtype([('TriggerNo', np.uint32), ('ChannelID', np.uint32), ('tscharge', np.float64), ('tswave', np.float64), ('mucharge', np.float64), ('muwave', np.float64)])
    ts = np.zeros(N, dtype=sdtp)
    ts['TriggerNo'] = ent['TriggerNo']
    ts['ChannelID'] = ent['ChannelID']
    with Pool(min(args.Ncpu, cpu_count())) as pool:
        result = pool.starmap(partial(fbmp_inference), slices)
    ts['tswave'] = np.hstack([result[i][0] for i in range(len(slices))])
    ts['tscharge'] = np.hstack([result[i][1] for i in range(len(slices))])
    As = np.hstack([result[i][2] for i in range(len(slices))])
    ts['muwave'] = np.hstack([result[i][3] for i in range(len(slices))])
    ts['mucharge'] = np.hstack([result[i][4] for i in range(len(slices))])
    d_tot = np.hstack([result[i][5] for i in range(len(slices))])
    d_max = np.hstack([result[i][6] for i in range(len(slices))])
    time_fbmp = np.hstack([result[i][7] for i in range(len(slices))]).mean()
    nu_truth = np.hstack([result[i][8] for i in range(len(slices))])
    nu_max = np.hstack([result[i][9] for i in range(len(slices))])
    num = np.hstack([result[i][10] for i in range(len(slices))])
    print('FBMP finished, real time {0:.02f}s'.format(time_fbmp))
    print('nu max larger than nu truth fraction is {0:.02%}'.format((nu_max > nu_truth).sum() / len(nu_truth)))

    matplotlib.use('Agg')
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    ff = plt.figure(figsize=(16, 24))
    # ff.tight_layout()
    ax = ff.add_subplot(321)
    # di, ci = np.unique(d_tot, return_counts=True)
    # ax.bar(di, ci, label='d_tot')
    ax.hist(d_tot, bins=np.linspace(0, d_tot.max(), 50), label='d_tot')
    ax.set_xlabel('d_tot')
    ax.set_ylabel('Count')
    ax.set_xlim(right=1.01 * d_tot.max())
    ax.set_ylim(0.5, N)
    ax.set_yscale('log')
    ax.legend(loc='upper center')
    ax = ff.add_subplot(322)
    # di, ci = np.unique(d_max, return_counts=True)
    # ax.bar(di, ci, label='d_max')
    ax.hist(d_max, bins=np.linspace(0, d_tot.max(), 50), label='d_max')
    ax.set_xlabel('d_max')
    ax.set_ylabel('Count')
    ax.set_xlim(right=1.01 * d_tot.max())
    ax.set_ylim(0.5, N)
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    ax = ff.add_subplot(323)
    ax.hist(d_max / d_tot, label='frac', range=(0, 1), bins=100)
    ax.set_xlabel('d_max / d_tot')
    ax.set_ylabel('Count')
    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, N)
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    ax = ff.add_subplot(324)
    ax.hist(nu_max - nu_truth, label=r'$\nu_{max}-\nu_{tru}$', bins=np.linspace((nu_max - nu_truth).min(), (nu_max - nu_truth).max(), 50))
    ax.set_xlabel(r'$\Delta\nu$')
    ax.set_ylabel('Count')
    ax.set_ylim(0.5, N)
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    ax = ff.add_subplot(325)
    ax.hist(num, bins=np.linspace(0, num.max(), 50), label=r'$N_{sam}$')
    ax.set_xlabel(r'$N_{sam}$')
    ax.set_ylabel('Count')
    ax.set_ylim(0.5, N)
    ax.set_yscale('log')
    m = num.mean()
    ax.legend(loc='upper right', title=fr'$E[N_{{sam}}]={m:.02f}$')
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