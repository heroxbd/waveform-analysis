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
from scipy.interpolate import interp1d
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
from matplotlib.colors import LogNorm
from numba import njit

import wf_func as wff

np.random.seed(7)

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
Thres = wff.Thres
mix0sigma = 1e-3
mu0 = np.arange(1, int(Mu + 5 * np.sqrt(Mu)))
n_t = np.arange(1, 20)
p_t = special.comb(mu0, 2)[:, None] * np.power(wff.convolve_exp_norm(np.arange(1029) - 200, Tau, Sigma) / n_t[:, None], 2).sum(axis=1)
n0 = np.array([n_t[p_t[i] < max(1e-1, np.sort(p_t[i])[1])].min() for i in range(len(mu0))])
ndict = dict(zip(mu0, n0))
TRIALS = wff.TRIALS

Δt_r = Δt_l = 0
while wff.log_convolve_exp_norm(Δt_l, Tau, Sigma) > np.log(1e-9):
    Δt_l -= 5
while wff.log_convolve_exp_norm(Δt_r, Tau, Sigma) > np.log(1e-9):
    Δt_r += 5

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
    Awindow = int(window * 0.95)
    rng_key = jax.random.PRNGKey(1)
    rng_key, rng_key_ = jax.random.split(rng_key)
    stime_t0 = np.empty(a1 - a0)
    stime_cha = np.empty(a1 - a0)
    accep = np.full(a1 - a0, np.nan)
    mix0ratio = np.full(a1 - a0, np.nan)
    dt = np.zeros((a1 - a0) * Awindow * 2, dtype=opdt)
    time_mcmc = np.empty(a1 - a0)
    start = 0
    end = 0
    count = 0
    b = [0., 600.]
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
        time_mcmc_start = time.time()
        truth = pelist[pelist['TriggerNo'] == ent[i]['TriggerNo']]
        cid = ent[i]['ChannelID']
        wave = ent[i]['Waveform'].astype(np.float64) * spe_pre[cid]['epulse']

        mu = abs(wave.sum() / gmu)
        n = ndict[min(math.ceil(mu), max(mu0))]
        AV, wave, tlist, t0_init, t0_init_delta, A_init, left_wave, right_wave = wff.initial_params(wave[::wff.nshannon], spe_pre[cid], Tau, Sigma, gmu, Thres['lucyddm'], p, is_t0=True, n=n, nshannon=1)
        mu = abs(wave.sum() / gmu)
        AV = jnp.array(AV)
        wave = jnp.array(wave)
        tlist = jnp.array(tlist)
        t0_init = jnp.array(t0_init)
        A_init = jnp.array(A_init)

        nuts_kernel = numpyro.infer.NUTS(model, adapt_step_size=True, init_strategy=numpyro.infer.initialization.init_to_value(values={'t0': t0_init, 'A': A_init}))
        mcmc = numpyro.infer.MCMC(nuts_kernel, num_samples=1000, num_warmup=1000, num_chains=1, progress_bar=False, chain_method='sequential', jit_model_args=True)
        try:
            ticrun = time.time()
            mcmc.run(rng_key, n=n, y=wave, mu=mu, tlist=tlist, AV=AV, t0left=t0_init - 3 * Sigma, t0right=t0_init + 3 * Sigma, extra_fields=('accept_prob', 'potential_energy'))
            tocrun = time.time()
            potential_energy = np.array(mcmc.get_extra_fields()['potential_energy'])
            accep[i - a0] = np.array(mcmc.get_extra_fields()['accept_prob']).mean()
            t0_t0 = np.array(mcmc.get_samples()['t0']).flatten()
            A = np.array(mcmc.get_samples()['A'])
            count = count + 1
        except:
            t0_t0 = np.array(t0_init)
            t0_cha = t0_init
            tlist = np.array(tlist)
            A = np.array([A_init])
            print('Failed waveform is TriggerNo = {:05d}, ChannelID = {:02d}, i = {:05d}'.format(ent[i]['TriggerNo'], cid, i))
        time_mcmc[i - a0] = time.time() - time_mcmc_start
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
    return stime_t0, stime_cha, time_mcmc, dt, count, accep, mix0ratio

n = 1
b_t0 = [0., 600.]

def fbmp_inference(a0, a1):
    t0_wav = np.empty(a1 - a0)
    t0_cha = np.empty(a1 - a0)
    mu_wav = np.empty(a1 - a0)
    mu_cha = np.empty(a1 - a0)
    mu_kl = np.empty(a1 - a0)
    time_fbmp = np.empty(a1 - a0)
    dt = np.zeros((a1 - a0) * window, dtype=opdt)
    d_max = np.zeros(a1 - a0).astype(int)
    elbo = np.zeros(a1 - a0)
    start = 0
    end = 0
    for i in range(a0, a1):
        time_fbmp_start = time.time()
        cid = ent[i]['ChannelID']
        assert cid == 0
        wave = ent[i]['Waveform'].astype(np.float64) * spe_pre[cid]['epulse']

        # initialization
        A, y, tlist, t0_t, t0_delta, cha, left_wave, right_wave = wff.initial_params(wave[::wff.nshannon], spe_pre[ent[i]['ChannelID']], Tau, Sigma, gmu, Thres['lucyddm'], p, is_t0=True, is_delta=False, n=n)
        # assert len(np.unique(np.diff(tlist))) == 1
        s_cha = np.cumsum(cha)
        # moving average filter of size 2*n+1
        cha = np.pad(s_cha[2*n+1:], (n+1, n), 'edge') - np.pad(s_cha[:-(2*n+1)], (n+1, n), 'edge')
        cha += 1e-8 # for completeness of the random walk.
        p_cha = cha / np.sum(cha)
        mu_t = abs(y.sum() / gmu)

        truth = pelist[pelist['TriggerNo'] == ent[i]['TriggerNo']]
        # t0_t = t0_truth['T0'][i] # override with truth to debug mu
        # tlist = truth['HitPosInWindow'][truth['HitPosInWindow'] < right_wave - 1]
        # t_auto = (np.arange(left_wave, right_wave) / wff.nshannon)[:, None] - tlist
        # A = wff.spe((t_auto + np.abs(t_auto)) / 2, p[0], p[1], p[2])

        # Eq. (9) where the columns of A are taken to be unit-norm.
        mus = np.sqrt(np.diag(np.matmul(A.T, A)))
        assert np.std(mus) < 1e-4, 'mus must be equal'
        mus = mus[0]
        A = A / mus
        '''
        A: basis dictionary
        p1: prior probability for each bin.
        sig2w: variance of white noise.
        sig2s: variance of signal x_i.
        mus: mean of signal x_i.
        TRIALS: number of Metropolis steps.
        '''
        p1 = mu_t * wff.convolve_exp_norm(tlist - t0_t, Tau, Sigma) / n + 1e-8
        # p1 = cha / cha.sum() * mu_t + 1e-8
        # p1 = p1 / p1.sum() * mu_t
        sig2w = spe_pre[cid]['std'] ** 2
        sig2s = (gsigma * mus / gmu) ** 2

        nu_star, T_star, c_star, es_history, NPE_evo = wff.metropolis_fbmp(y, A, sig2w, sig2s, mus, p1, p_cha, mu_t)
        time_fbmp[i - a0] = time.time() - time_fbmp_start
        num = len(nu_star)

        # Extra calculation to test
        # elbo_i = 0
        p1_truth = Mu * wff.convolve_exp_norm(tlist - t0_truth[i]['T0'], Tau, Sigma) / n + 1e-8
        nu_space_prior = np.array([wff.nu_direct(y, A, c_star[j], mus, sig2s, sig2w, p1_truth) for j in range(num)])
        elbo_i = wff.elbo(nu_space_prior)

        ilp_cha = np.log(cha.sum()) - np.log(cha)
        guess = ilp_cha[es_history['loc'].astype(int)]
        es_history['loc'] = np.interp(es_history['loc'], xp=np.arange(0.5, len(tlist)), fp=tlist)
        ans = opti.fmin_l_bfgs_b(lambda x: -np.sum(wff.log_convolve_exp_norm(es_history['loc'] - x, Tau, Sigma)), x0=[t0_t], approx_grad=True, bounds=[b_t0], maxfun=500000)
        t00 = ans[0].item() if ans[-1]['warnflag'] == 0 else t0_t
        mu = mu_t
        b_mu = [max(1e-8, mu - 5 * np.sqrt(mu)), mu + 5 * np.sqrt(mu)]
        def agg_NPE(t0):
            log_f = wff.log_convolve_exp_norm(es_history['loc'] - t0, Tau, Sigma) + guess
            return wff.jit_agg_NPE(es_history['step'], log_f, TRIALS)

        def t_t0(t0):
            nonlocal mu
            NPE, f_agg = agg_NPE(t0)
            ans = opti.fmin_l_bfgs_b(lambda μ: μ - special.logsumexp(NPE * np.log(μ / mu) + f_agg), x0=[mu], approx_grad=True, bounds=[b_mu], maxfun=500000)
            mu = ans[0].item()
            return ans[1]

        ans = opti.fmin_l_bfgs_b(t_t0, x0=[t00], approx_grad=True, bounds=[b_t0], maxfun=500000)
        t0 = ans[0].item()

        j = 0
        xmmse_most = np.zeros(len(tlist))
        while np.all(xmmse_most <= 0):
            maxindex = nu_star.argsort()[::-1][j]
            zx = y - np.dot(A, mus * c_star[maxindex])
            Phi_s = wff.Phi(y, A, c_star[maxindex], mus, sig2s, sig2w)
            invPhi = np.linalg.inv(Phi_s)
            xmmse_most = mus * c_star[maxindex] + np.matmul(np.diagflat(sig2s * c_star[maxindex]), np.matmul(A.T, np.matmul(invPhi, zx)))
            j += 0
        pet = np.repeat(tlist[xmmse_most > 0], c_star[maxindex][xmmse_most > 0])
        cha = np.repeat(xmmse_most[xmmse_most > 0] / mus / c_star[maxindex][xmmse_most > 0], c_star[maxindex][xmmse_most > 0])
        mu_i = (c_star[maxindex] > 0).sum()
        t0_i, _ = wff.likelihoodt0(pet, char=cha, gmu=gmu, Tau=Tau, Sigma=Sigma, mode='all')
        pet, cha = wff.clip(pet, cha, Thres[method])
        cha = cha * gmu
        d_max[i - a0] = maxindex
        elbo[i - a0] = elbo_i
        t0_wav[i - a0] = t0
        t0_cha[i - a0] = t0_i
        mu_wav[i - a0] = mu
        mu_cha[i - a0] = mu_i
        mu_kl[i - a0] = cha.sum()
        end = start + len(cha)
        dt['HitPosInWindow'][start:end] = pet
        dt['Charge'][start:end] = cha
        dt['TriggerNo'][start:end] = ent[i]['TriggerNo']
        dt['ChannelID'][start:end] = ent[i]['ChannelID']
        start = end
    dt = dt[:end]
    dt = np.sort(dt, kind='stable', order=['TriggerNo', 'ChannelID'])
    return t0_wav, t0_cha, dt, mu_wav, mu_cha, mu_kl, time_fbmp, elbo, d_max

print('Initialization finished, real time {0:.02f}s, cpu time {1:.02f}s'.format(time.time() - global_start, time.process_time() - cpu_global_start))
tic = time.time()
cpu_tic = time.process_time()

ent = np.sort(ent, kind='stable', order=['TriggerNo', 'ChannelID'])
Chnum = len(np.unique(ent['ChannelID']))
e_pel = ent['TriggerNo'] * Chnum + ent['ChannelID']
e_pel, i_pel = np.unique(e_pel, return_index=True)
i_pel = np.append(i_pel, len(ent))

opdt = np.dtype([('TriggerNo', np.uint32), ('ChannelID', np.uint32), ('HitPosInWindow', np.float64), ('Charge', np.float64)])

# N = 1000
if args.Ncpu == 1:
    slices = [[0, N]]
else:
    chunk = N // args.Ncpu + 1
    slices = np.vstack((np.arange(0, N, chunk), np.append(np.arange(chunk, N, chunk), N))).T.astype(int).tolist()

if method == 'mcmc':
    sdtp = np.dtype([('TriggerNo', np.uint32), ('ChannelID', np.uint32), ('tscharge', np.float64), ('tswave', np.float64), ('consumption', np.float64)])
    ts = np.zeros(N, dtype=sdtp)
    ts['TriggerNo'] = ent['TriggerNo']
    ts['ChannelID'] = ent['ChannelID']
    with Pool(min(args.Ncpu, cpu_count())) as pool:
        result = pool.starmap(partial(time_numpyro), slices)
    ts['tswave'] = np.hstack([result[i][0] for i in range(len(slices))])
    ts['tscharge'] = np.hstack([result[i][1] for i in range(len(slices))])
    ts['consumption'] = np.hstack([result[i][2] for i in range(len(slices))])
    print('MCMC finished, real time {0:.02f}s'.format(ts['consumption'].sum()))
    dt = np.hstack([result[i][3] for i in range(len(slices))])
    count = np.sum([result[i][4] for i in range(len(slices))])
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

    dt = np.sort(dt, kind='stable', order=['TriggerNo', 'ChannelID'])
    print('Successful MCMC ratio is {:.4%}'.format(count / N))
elif method == 'fbmp':
    sdtp = np.dtype([('TriggerNo', np.uint32), ('ChannelID', np.uint32), ('tscharge', np.float64), ('tswave', np.float64), ('mucharge', np.float64), ('muwave', np.float64), ('mukl', np.float64), ('elbo', np.float64), ('consumption', np.float64)])
    ts = np.zeros(N, dtype=sdtp)
    ts['TriggerNo'] = ent['TriggerNo'][:N]
    ts['ChannelID'] = ent['ChannelID'][:N]
    # fbmp_inference(8363, 8400)
    with Pool(min(args.Ncpu, cpu_count())) as pool:
        result = pool.starmap(partial(fbmp_inference), slices)
    ts['tswave'] = np.hstack([result[i][0] for i in range(len(slices))])
    ts['tscharge'] = np.hstack([result[i][1] for i in range(len(slices))])
    dt = np.hstack([result[i][2] for i in range(len(slices))])
    ts['muwave'] = np.hstack([result[i][3] for i in range(len(slices))])
    ts['mucharge'] = np.hstack([result[i][4] for i in range(len(slices))])
    ts['mukl'] = np.hstack([result[i][5] for i in range(len(slices))])
    ts['consumption'] = np.hstack([result[i][6] for i in range(len(slices))])
    print('FBMP finished, real time {0:.02f}s'.format(ts['consumption'].sum()))
    ts['elbo'] = np.hstack([result[i][7] for i in range(len(slices))])
    d_max = np.hstack([result[i][8] for i in range(len(slices))])
    
    N_add = N / (1 - poisson.cdf(0, Mu)) * poisson.cdf(0, Mu)
    print('relative bias is {:.3%}'.format((np.mean(np.append(ts['muwave'], np.zeros(round(N_add)))) - Mu) / Mu))

    matplotlib.use('Agg')
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    ff = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 2, figure=ff, left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.3, hspace=0.3)
    # ff.tight_layout()
    ax = ff.add_subplot(gs[0, 0])
    ax.hist(d_max, label=r'$N_{max}$', bins=np.arange(d_max.max() + 1))
    ax.set_xlabel(r'$N_{max}$')
    ax.set_ylabel('Count')
    ax.set_xlim(0, d_max.max())
    ax.set_ylim(0.5, N)
    ax.set_yscale('log')
    m = d_max.mean()
    ax.legend(loc='upper right', title=fr'$E[N_{{max}}]={m:.02f}$')
    ax = ff.add_subplot(gs[0, 1])
    ax.hist(ts['elbo'], bins=np.linspace(ts['elbo'].min(), ts['elbo'].max(), 51), label=r'$\mathrm{ELBO}$')
    ax.set_xlabel(r'$\mathrm{ELBO}$')
    ax.set_ylabel('Count')
    ax.set_ylim(0.5, N)
    ax.set_yscale('log')
    m = ts['elbo'].mean()
    ax.legend(loc='upper right', title=fr'$E[\mathrm{{ELBO}}]={m:.02f}$')
    ff.savefig(os.path.splitext(fopt)[0] + '.png')
    plt.close()

    dt = np.sort(dt, kind='stable', order=['TriggerNo', 'ChannelID'])

print('Prediction generated, real time {0:.02f}s, cpu time {1:.02f}s'.format(time.time() - tic, time.process_time() - cpu_tic))

with h5py.File(fopt, 'w') as opt:
    pedset = opt.create_dataset('photoelectron', data=dt, compression='gzip')
    pedset.attrs['Method'] = method
    pedset.attrs['mu'] = Mu
    pedset.attrs['tau'] = Tau
    pedset.attrs['sigma'] = Sigma
    tsdset = opt.create_dataset('starttime', data=ts, compression='gzip')
    print('The output file path is {}'.format(fopt))

print('Finished! Consuming {0:.02f}s in total, cpu time {1:.02f}s.'.format(time.time() - global_start, time.process_time() - cpu_global_start))
