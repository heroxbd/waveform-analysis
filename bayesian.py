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
import numdifftools as nd

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
Thres = wff.Thres
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

prior = True
space = True
n = 2
tlist_pan = np.sort(np.unique(np.hstack(np.arange(0, window)[:, None] + np.linspace(0, 1, n, endpoint=False) - (n // 2) / n)))
b_t0 = [0., 600.]

def likelihood(mu, t0, As_k, nu_star_k):
    a = wff.convolve_exp_norm(tlist_pan - t0, Tau, Sigma) / n + 1e-8 # use tlist_pan not tlist
    # a *= mu / a.sum()
    a *= mu
    li = -special.logsumexp(np.log(poisson.pmf(As_k, mu=a)).sum(axis=1) + nu_star_k)
    return li

def sum_mu_likelihood(mu, t0_list, As_list, nu_star_list):
    return np.sum([likelihood(mu, t0_list[k], As_list[k], nu_star_list[k]) for k in range(len(t0_list))])

def optit0mu(mu, t0, nu_star, As, mu_init=None):
    l = len(t0)
    mulist = np.arange(max(1e-8, mu - 2 * np.sqrt(mu)), mu + 2 * np.sqrt(mu), 1e-1)
    b_mu = [max(1e-8, mu - 5 * np.sqrt(mu)), mu + 5 * np.sqrt(mu)]
    # psy_star = [np.exp(nu_star[k] - nu_star[k].max()) / np.sum(np.exp(nu_star[k] - nu_star[k].max())) for k in range(l)]
    t0list = [np.arange(t0[k] - 3 * Sigma, t0[k] + 3 * Sigma + 1e-6, 0.2) for k in range(l)]
    sigmamu = None
    logLv_mu = None
    if mu_init is None:
        mu_init = np.empty(l)
        for k in range(l):
            mu_init[k] = mulist[np.array([likelihood(mulist[j], t0[k], As[k], nu_star[k]) for j in range(len(mulist))]).argmin()]
            t0_init = t0list[k][np.array([likelihood(mu_init[k], t0list[k][j], As[k], nu_star[k]) for j in range(len(t0list[k]))]).argmin()]
            likelihood_x = lambda x, As, nu_star: likelihood(x[0], x[1], As, nu_star)
            t0[k] = opti.fmin_l_bfgs_b(likelihood_x, args=(As[k], nu_star[k]), x0=[mu_init[k], t0_init], approx_grad=True, bounds=[b_mu, b_t0], maxfun=50000)[0][1]
        Likelihood = lambda mu: np.sum([likelihood(mu, t0[k], As[k], nu_star[k]) for k in range(l)])
        mu, fval, _ = opti.fmin_l_bfgs_b(Likelihood, x0=[np.mean(mu_init)], approx_grad=True, bounds=[b_mu], maxfun=50000)
    else:
        def Likelihood(mu, t0_list, As_list, nu_star_list):
            with Pool(min(args.Ncpu // 3, cpu_count())) as pool:
                result = np.sum(pool.starmap(likelihood, zip([mu] * l, t0_list, As_list, nu_star_list)))
            return result
        mu, fval, _ = opti.fmin_l_bfgs_b(Likelihood, args=[t0, As, nu_star], x0=[np.mean(mu_init)], approx_grad=True, bounds=[b_mu], maxfun=50000, factr=100.0, pgtol=1e-10)
        print('Mu fitting info is', _)
        sigmamu_est = np.sqrt(mu / N)
        mulist = np.sort(np.append(np.arange(max(1e-8, mu - 2 * sigmamu_est), mu + 2 * sigmamu_est, sigmamu_est / 50), mu))
        partial_sum_mu_likelihood = partial(sum_mu_likelihood, t0_list=t0, As_list=As, nu_star_list=nu_star)
        with Pool(min(args.Ncpu // 3, cpu_count())) as pool:
            logLv_mu = np.array(pool.starmap(partial_sum_mu_likelihood, zip(mulist)))

        mu_func = interp1d(mulist, logLv_mu, bounds_error=False, fill_value='extrapolate')
        logLvdelta = np.vectorize(lambda mu_t: np.abs(mu_func(mu_t) - fval - 0.5))
        # sigmamu = abs(opti.fmin_l_bfgs_b(logLvdelta, x0=[mulist[np.abs(logLv_mu - fval - 0.5).argmin()]], approx_grad=True, bounds=[[mulist[0], mulist[-1]]], maxfun=500000)[0] - mu) * np.sqrt(l)
        sigmamu_l = abs(opti.fmin_l_bfgs_b(logLvdelta, x0=[mulist[mulist <= mu][np.abs(logLv_mu[mulist <= mu] - fval - 0.5).argmin()]], approx_grad=True, bounds=[[mulist[0], mulist[-1]]], maxfun=500000)[0] - mu) * np.sqrt(l)
        sigmamu_r = abs(opti.fmin_l_bfgs_b(logLvdelta, x0=[mulist[mulist > mu][np.abs(logLv_mu[mulist > mu] - fval - 0.5).argmin()]], approx_grad=True, bounds=[[mulist[0], mulist[-1]]], maxfun=500000)[0] - mu) * np.sqrt(l)
        sigmamu = (sigmamu_l + sigmamu_r) / 2
        print('Finite difference sigmamu is {:.4}'.format(sigmamu.item()))

        # derivative_mu2 = nd.Derivative(Likelihood, step=1e-10, n=2, full_output=True)
        # s, info = derivative_mu2(mu, t0_list=t0, As_list=As, nu_star_list=nu_star)
        # print('Mu derivative info is', info)
        # sigmamu = 1 / np.sqrt(s) * np.sqrt(l)
    return mu, t0, [sigmamu, mulist, logLv_mu, fval]

def fbmp_inference(a0, a1):
    t0_wav = np.empty(a1 - a0)
    t0_cha = np.empty(a1 - a0)
    mu_wav = np.empty(a1 - a0)
    mu_cha = np.empty(a1 - a0)
    mu_kl = np.empty(a1 - a0)
    time_fbmp = np.empty(a1 - a0)
    dt = np.zeros((a1 - a0) * window, dtype=opdt)
    d_tot = np.zeros(a1 - a0).astype(int)
    d_max = np.zeros(a1 - a0).astype(int)
    elbo = np.zeros(a1 - a0).astype(int)
    nu_truth = np.empty(a1 - a0)
    nu_max = np.empty(a1 - a0)
    num = np.zeros(a1 - a0).astype(int)
    nu_star_list = []
    As_list = []
    start = 0
    end = 0
    for i in range(a0, a1):
        time_fbmp_start = time.time()
        cid = ent[i]['ChannelID']
        assert cid == 0
        wave = ent[i]['Waveform'].astype(np.float64) * spe_pre[cid]['epulse']

        # initialization
        mu_t = abs(wave.sum() / gmu)
        A, y, tlist, t0_t, t0_delta, cha, left_wave, right_wave = wff.initial_params(wave[::wff.nshannon], spe_pre[ent[i]['ChannelID']], Tau, Sigma, gmu, Thres['lucyddm'], p, is_t0=True, is_delta=False, n=n, nshannon=1)
        mu_t = abs(y.sum() / gmu)

        truth = pelist[pelist['TriggerNo'] == ent[i]['TriggerNo']]

        # tlist = truth['HitPosInWindow'][truth['HitPosInWindow'] < right_wave - 1]
        # t_auto = (np.arange(left_wave, right_wave) / wff.nshannon)[:, None] - tlist
        # A = wff.spe((t_auto + np.abs(t_auto)) / 2, p[0], p[1], p[2])

        # Eq. (9) where the columns of A are taken to be unit-norm.
        mus = np.sqrt(np.diag(np.matmul(A.T, A)))
        A = A / mus
        p1 = mu_t * wff.convolve_exp_norm(tlist - t0_t, Tau, Sigma) / n + 1e-8
        # p1 = cha / cha.sum() * mu_t + 1e-8
        p1 = p1 / p1.sum() * mu_t
        sig2w = spe_pre[cid]['std'] ** 2
        sig2s = (gsigma * mus / gmu) ** 2
        xmmse_star, nu_star, T_star, c_star, d_max_i, num_i = wff.fbmpr_fxn_reduced(y, A, sig2w, sig2s, mus, len(p1), p1=p1, truth=truth, i=i, left=left_wave, right=right_wave, tlist=tlist, gmu=gmu, para=p, prior=prior, space=space)
        time_fbmp[i - a0] = time.time() - time_fbmp_start

        la_truth = Mu * wff.convolve_exp_norm(tlist - t0_truth[i]['T0'], Tau, Sigma) / n + 1e-8
        nu_space_prior = np.array([wff.nu_direct(y, A, c_star[j], mus, sig2s, sig2w, la_truth, prior=True, space=True) for j in range(num_i)])

        nx = np.sum([(tlist - 0.5 / n <= truth['HitPosInWindow'][j]) * (tlist + 0.5 / n > truth['HitPosInWindow'][j]) for j in range(len(truth))], axis=0)
        cc = np.sum([np.where(tlist - 0.5 / n < truth['HitPosInWindow'][j], np.sqrt(truth['Charge'][j]), 0) * np.where(tlist + 0.5 / n > truth['HitPosInWindow'][j], np.sqrt(truth['Charge'][j]), 0) for j in range(len(truth))], axis=0)
        pp = np.arange(left_wave, right_wave)
        wav_ans = np.sum([np.where(pp > truth['HitPosInWindow'][j], wff.spe(pp - truth['HitPosInWindow'][j], tau=p[0], sigma=p[1], A=p[2]) * truth['Charge'][j] / gmu, 0) for j in range(len(truth))], axis=0)
        nu_truth[i - a0] = wff.nu_direct(y, A, nx, mus, sig2s, sig2w, p1, prior=prior, space=space)
        nu = np.array([wff.nu_direct(y, A, c_star[j], mus, sig2s, sig2w, p1, prior=prior, space=space) for j in range(num_i)])
        assert abs(nu - nu_star).max() < 1e-6
        nu_max[i - a0] = nu[0]
        rss_truth = np.power(wav_ans - np.matmul(A, cc / gmu * mus), 2).sum()
        rss = np.array([np.power(wav_ans - np.matmul(A, xmmse_star[j]), 2).sum() for j in range(num_i)])

        maxindex = nu_star.argmax()
        xmmse_most = np.clip(xmmse_star[maxindex], 0, np.inf)
        pet = np.repeat(tlist[xmmse_most > 0], c_star[maxindex][xmmse_most > 0])
        cha = np.repeat(xmmse_most[xmmse_most > 0] / mus[xmmse_most > 0] / c_star[maxindex][xmmse_most > 0], c_star[maxindex][xmmse_most > 0])

        # c_star_truth = np.sum([np.where(tlist - 0.5 / n < truth['HitPosInWindow'][j], 1, 0) * np.where(tlist + 0.5 / n > truth['HitPosInWindow'][j], 1, 0) for j in range(len(truth))], axis=0)
        # if c_star_truth.sum() == 0:
        #     c_star_truth[0] = 1
        # As_truth = np.zeros((1, len(tlist_pan)))
        # As_truth[:, np.isin(tlist_pan, tlist)] = c_star_truth[None, :]
        # assert sum(np.sum(As_truth, axis=0) > 0) > 0
        # mu, t0, _ = optit0mu(len(truth), [t0_truth[i]['T0']], [np.array([0.])], [As_truth], [p1])
        As = np.zeros((num_i, len(tlist_pan)))
        As[:, np.isin(tlist_pan, tlist)] = c_star
        assert sum(np.sum(As, axis=0) > 0) > 0

        spacefactor = 0
        priorfactor = 0
        if not space:
            spacefactor = np.array([-0.5 * np.log(np.linalg.det(wff.Phi(y, A, c_star[j], mus, sig2s, sig2w, p1))) for j in range(num_i)])
        if prior:
            priorfactor = -poisson.logpmf(c_star, p1).sum(axis=1)
        nu_star = nu_star + priorfactor + spacefactor
        # nu_star = np.log(np.exp(nu_star - nu_star.max()) / np.sum(np.exp(nu_star - nu_star.max())))
        mu, t0, _ = optit0mu(mu_t, [t0_t], [nu_star], [As])
        mu_i, t0_i, _ = optit0mu(mu_t, [t0_t], [np.array([0.])], [As[maxindex][None, :]])

        d_tot[i - a0] = len(p1)
        d_max[i - a0] = d_max_i
        elbo[i - a0] = wff.elbo(nu_space_prior)
        pet, cha = wff.clip(pet, cha, Thres[method])
        cha = cha * gmu
        t0_wav[i - a0] = t0[0]
        t0_cha[i - a0] = t0_i[0]
        mu_wav[i - a0] = mu
        mu_cha[i - a0] = mu_i
        mu_kl[i - a0] = cha.sum()
        num[i - a0] = num_i
        end = start + len(cha)
        dt['HitPosInWindow'][start:end] = pet
        dt['Charge'][start:end] = cha
        dt['TriggerNo'][start:end] = ent[i]['TriggerNo']
        dt['ChannelID'][start:end] = ent[i]['ChannelID']
        start = end
        nu_star_list.append(nu_star)
        As_list.append(As)
    dt = dt[:end]
    dt = np.sort(dt, kind='stable', order=['TriggerNo', 'ChannelID'])
    return t0_wav, t0_cha, dt, mu_wav, mu_cha, mu_kl, time_fbmp, d_tot, d_max, nu_truth, nu_max, num, elbo, nu_star_list, As_list

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
    # fbmp_inference(0, 100)
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
    d_tot = np.hstack([result[i][7] for i in range(len(slices))])
    d_max = np.hstack([result[i][8] for i in range(len(slices))])
    nu_truth = np.hstack([result[i][9] for i in range(len(slices))])
    nu_max = np.hstack([result[i][10] for i in range(len(slices))])
    num = np.hstack([result[i][11] for i in range(len(slices))])
    ts['elbo'] = np.hstack([result[i][12] for i in range(len(slices))])
    nu_star_list = []
    for i in range(len(slices)):
        nu_star_list += result[i][13]
    As_list = []
    for i in range(len(slices)):
        As_list += result[i][14]
    N_add = round(N / (1 - poisson.cdf(0, Mu)) * poisson.cdf(0, Mu))
    nu_star_list_add = [np.array([0.])] * N_add
    nu_star_list += nu_star_list_add
    As_list_add = [np.zeros((1, len(tlist_pan)))] * N_add
    As_list += As_list_add
    t0_init = np.append(ts['tswave'], np.ones(N_add) * 100)
    assert np.all(np.array([len(t0_init), len(nu_star_list), len(As_list)]) == N + N_add)

    # mu, t0, sigmamu_list = optit0mu(Mu, t0_init, nu_star_list, As_list, mu_init=ts['muwave'])
    mu = np.array([np.nan])
    sigmamu_list = [np.array([np.nan]), np.nan, np.nan, np.nan]
    sigmamu = sigmamu_list[0]

    print('mu is {0:.4f}, sigma_mu is {1:.4f}'.format(mu.item(), sigmamu.item()))
    print('nu max larger than nu truth fraction is {0:.02%}'.format((nu_max > nu_truth).sum() / len(nu_truth)))

    matplotlib.use('Agg')
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    ff = plt.figure(figsize=(16, 18))
    gs = gridspec.GridSpec(3, 2, figure=ff, left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.3, hspace=0.3)
    # ff.tight_layout()
    ax = ff.add_subplot(gs[0, 0])
    h = ax.hist2d(d_tot, d_max, bins=(np.linspace(0, d_tot.max(), 51), np.linspace(0, d_tot.max(), 51)), norm=LogNorm())
    ff.colorbar(h[3], ax=ax, aspect=50)
    ax.set_xlabel('d_tot')
    ax.set_ylabel('d_max')
    ax = ff.add_subplot(gs[0, 1])
    ax.plot(sigmamu_list[1], 2 * (sigmamu_list[2] - sigmamu_list[3]), label=r'$-2\Delta\log L$')
    ax.axvline(x=mu, color='r')
    ax.axhline(y=1, color='k', linestyle='dashed', alpha=0.5)
    ax.axhline(y=0, color='k')
    ax.grid()
    ax.set_xlabel('mu')
    ax.set_ylim(-0.1, 4)
    ax.set_ylabel(r'$-2\Delta\log L$')
    ax.legend(loc='upper right')
    ax = ff.add_subplot(gs[1, 0])
    ax.hist(d_max / d_tot, label='frac', range=(0, 1), bins=100)
    ax.set_xlabel('d_max / d_tot')
    ax.set_ylabel('Count')
    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, N)
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    ax = ff.add_subplot(gs[1, 1])
    ax.hist(nu_max - nu_truth, label=r'$\nu_{max}-\nu_{tru}$', bins=np.linspace((nu_max - nu_truth).min(), (nu_max - nu_truth).max(), 51))
    ax.set_xlabel(r'$\Delta\nu$')
    ax.set_ylabel('Count')
    ax.set_ylim(0.5, N)
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    ax = ff.add_subplot(gs[2, 0])
    ax.hist(num, bins=np.linspace(0, num.max(), 51), label=r'$N_{sam}$')
    ax.set_xlabel(r'$N_{sam}$')
    ax.set_ylabel('Count')
    ax.set_ylim(0.5, N)
    ax.set_yscale('log')
    m = num.mean()
    ax.legend(loc='upper right', title=fr'$E[N_{{sam}}]={m:.02f}$')
    ax = ff.add_subplot(gs[2, 1])
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
    if method == 'fbmp':
        tsdset.attrs['mu'] = mu
        tsdset.attrs['sigmamu'] = sigmamu
    print('The output file path is {}'.format(fopt))

print('Finished! Consuming {0:.02f}s in total, cpu time {1:.02f}s.'.format(time.time() - global_start, time.process_time() - cpu_global_start))
