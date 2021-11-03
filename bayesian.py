import os
import sys
import re
import time
import math
import argparse
import pickle
from functools import partial
from multiprocessing import Pool, cpu_count
import itertools as it

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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm

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
    PEList = ipt['SimTriggerInfo/PEList'][:]
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

def sum_mu_likelihood(mu, t0_list, As_list, psy_star_list):
    return np.sum([likelihood(mu, t0_list[k], As_list[k], psy_star_list[k]) for k in range(len(t0_list))])

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
        def Likelihood(mu, t0_list, As_list, psy_star_list):
            result = np.sum(it.starmap(likelihood, zip([mu] * l, t0_list, As_list, psy_star_list)))
            return result
        mu, fval, _ = opti.fmin_l_bfgs_b(Likelihood, args=[t0, As, nu_star], x0=[np.mean(mu_init)], approx_grad=True, bounds=[b_mu], maxfun=50000)
        sigmamu_est = np.sqrt(mu / N)
        mulist = np.sort(np.append(np.arange(max(1e-8, mu - 2 * sigmamu_est), mu + 2 * sigmamu_est, sigmamu_est / 50), mu))
        partial_sum_mu_likelihood = partial(sum_mu_likelihood, t0_list=t0, As_list=As, psy_star_list=nu_star)
        logLv_mu = np.array(it.starmap(partial_sum_mu_likelihood, zip(mulist)))
        mu_func = interp1d(mulist, logLv_mu, bounds_error=False, fill_value='extrapolate')
        logLvdelta = np.vectorize(lambda mu_t: np.abs(mu_func(mu_t) - fval - 0.5))
        sigmamu = abs(opti.fmin_l_bfgs_b(logLvdelta, x0=[mulist[np.abs(logLv_mu - fval - 0.5).argmin()]], approx_grad=True, bounds=[[mulist[0], mulist[-1]]], maxfun=500000)[0] - mu) * np.sqrt(l)
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
    NPE_list = []
    avrNPE_list = []
    mu_t_list = []
    As_list = []
    start = 0
    end = 0
    for i in range(a0, a1):
        time_fbmp_start = time.time()
        eid = ent[i]["TriggerNo"]
        cid = ent[i]['ChannelID']
        assert cid == 0
        PEs = PEList[np.logical_and(PEList["TriggerNo"] == eid, PEList["PMTId"] == cid)]
        NPE_t = len(PEs)

        wave = ent[i]['Waveform'].astype(np.float64) * spe_pre[cid]['epulse']

        # initialization
        A, y, tlist, t0_t, t0_delta, cha, left_wave, right_wave = wff.initial_params(wave[::wff.nshannon], spe_pre[ent[i]['ChannelID']], Tau, Sigma, gmu, Thres['lucyddm'], p, is_t0=True, is_delta=False, n=n, nshannon=1)
        t0_t = t0_truth['T0'][i] # override with truth to debug mu
        mu_t = abs(y.sum() / gmu)
        # Eq. (9) where the columns of A are taken to be unit-norm.
        mus = np.sqrt(np.diag(np.matmul(A.T, A)))
        A = A / mus

        truth = pelist[pelist['TriggerNo'] == ent[i]['TriggerNo']]

        p1 = mu_t * wff.convolve_exp_norm(tlist - t0_t, Tau, Sigma) / n

        sig2w = spe_pre[cid]['std'] ** 2
        sig2s = (gsigma * mus / gmu) ** 2
        freq, NPE = wff.fbmpr_fxn_reduced(y, A, sig2w, sig2s, mus, p1)

        loggN = -NPE * np.log(mu_t) + np.log(freq)
        rst = opti.minimize_scalar(lambda μ: μ - special.logsumexp(loggN + NPE * np.log(μ)),
                                   bounds=(NPE[0], NPE[-1]))
        
        
        assert rst.success
        As_list.append(rst.x)
        mu_t_list.append(mu_t)
        NPE_list.append(NPE_t)
        avrNPE_list.append(np.average(NPE, weights=freq))

    return avrNPE_list, mu_t_list, As_list, NPE_list

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

if method == 'fbmp':
    sdtp = np.dtype([('TriggerNo', np.uint32), ('ChannelID', np.uint32), ('mucharge', np.float64), ('muwave', np.float64), ('avrNPE', np.float64), ("NPE_t", np.float64)])
    ts = np.zeros(N, dtype=sdtp)
    ts['TriggerNo'] = ent['TriggerNo'][:N]
    ts['ChannelID'] = ent['ChannelID'][:N]
    result = fbmp_inference(*slices[0])
    ts['avrNPE'] = result[0]
    ts['NPE_t'] = result[3]
    ts['mucharge'] = result[1]
    ts['muwave'] = result[2]
    dt = ts

print('Prediction generated, real time {0:.02f}s, cpu time {1:.02f}s'.format(time.time() - tic, time.process_time() - cpu_tic))

with h5py.File(fopt, 'w') as opt:
    pedset = opt.create_dataset('photoelectron', data=dt, compression='gzip')
    pedset.attrs['Method'] = method
    pedset.attrs['mu'] = Mu
    pedset.attrs['tau'] = Tau
    pedset.attrs['sigma'] = Sigma
    print('The output file path is {}'.format(fopt))

print('Finished! Consuming {0:.02f}s in total, cpu time {1:.02f}s.'.format(time.time() - global_start, time.process_time() - cpu_global_start))
