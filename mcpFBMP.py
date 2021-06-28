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
import mcp
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
mix0sigma = 1e-3
mu0 = np.arange(1, int(Mu + 5 * np.sqrt(Mu)))
n_t = np.arange(1, 100)
p_t = special.comb(mu0, 2)[:, None] * np.power(wff.convolve_exp_norm(np.arange(1029) - 200, Tau, Sigma) / n_t[:, None], 2).sum(axis=1)
n0 = np.array([n_t[p_t[i] < max(1e-2, np.sort(p_t[i])[1])].min() for i in range(len(mu0))])
ndict = dict(zip(mu0, n0))


D = 100
def fbmp_inference(a0, a1):
    nsp = 4
    nstd = 3
    t0_wav = np.empty(a1 - a0)
    t0_cha = np.empty(a1 - a0)
    mu_wav = np.empty(a1 - a0)
    mu_cha = np.empty(a1 - a0)
    dt = np.zeros((a1 - a0) * window, dtype=opdt)
    fbmpt0mu = np.zeros((a1-a0),dtype=[('t0','<f4'),('mu','<f4'),('fun','<f4'),('success','<i1'),('truthfun','<f4'),('trutht0','<f4'),('truthmu','<f4')])
    d_tot = np.zeros(a1 - a0).astype(int)
    d_max = np.zeros(a1 - a0).astype(int)
    start = 0
    end = 0
    b_t0 = [0., 600.]
    time_fbmp = 0
    for i in range(a0, a1):
        cid = ent[i]['ChannelID']
        wave = ent[i]['Waveform'].astype(np.float64) * spe_pre[cid]['epulse']

        # initialization
        mu_t = abs(wave.sum() / gmu/np.sum(mcp.normalTplProbability*np.arange(1,mcp.normalTplProbability.shape[0]+1)))
        # n = ndict[min(math.ceil(mu_t), max(mu0))]
        # n = math.ceil(max((mu_t + 3 * np.sqrt(mu_t)) * wff.convolve_exp_norm(np.arange(-3 * Sigma, 3 * Sigma + 2 * Tau, 0.1), Tau, Sigma)) * 4)
        n = 10
        A, wave_r, tlist, t0_t, t0_delta, cha, left_wave, right_wave = wff.initial_params(wave[::wff.nshannon], spe_pre[ent[i]['ChannelID']], Mu, Tau, Sigma, gmu, Thres['lucyddm'], p, nsp, nstd, is_t0=True, is_delta=False, n=n, nshannon=1)
        
        def optit0mu(t0, mu, n, xmmse_star, psy_star, c_star, la):
            # condition probability
            ys = np.log(psy_star) - np.log(poisson.pmf(c_star, la)).sum(axis=1)
            ys = np.exp(ys - ys.max()) / np.sum(np.exp(ys - ys.max()))
            btlist = np.arange(t0 - 3 * Sigma, t0 + 3 * Sigma + 1e-6, 0.2)
            mulist = np.arange(max(0.2, mu - 3 * np.sqrt(mu)), mu + 3 * np.sqrt(mu), 0.2)
            b_mu = [max(0.2, mu - 5 * np.sqrt(mu)), mu + 5 * np.sqrt(mu)]
            tlist_pan = np.sort(np.unique(np.hstack(np.arange(0, window)[:, None] + np.arange(0, 1, 1 / n))))
            As = np.zeros((len(xmmse_star), len(tlist_pan)))
            As[:, np.isin(tlist_pan, tlist)] = c_star
            assert sum(np.sum(As, axis=0) > 0) > 0
            
            # optimize t0
            logL = lambda t0 : -1 * np.sum(special.logsumexp((np.log(wff.convolve_exp_norm(tlist_pan - t0, Tau, Sigma) + 1e-8)[None, :] * As).sum(axis=1), b=ys))
            logLv_btlist = np.vectorize(logL)(btlist)
            t0 = opti.fmin_l_bfgs_b(logL, x0=[btlist[np.argmin(logLv_btlist)]], approx_grad=True, bounds=[b_t0], maxfun=50000)[0]

            # optimize mu
            def likelihood(mu):
                a = mu * wff.convolve_exp_norm(tlist_pan - t0, Tau, Sigma) / n + 1e-8 # use tlist_pan not tlist
                li = -special.logsumexp(np.log(poisson.pmf(As, a)).sum(axis=1), b=ys)
                return li
            like = np.array([likelihood(mulist[j]) for j in range(len(mulist))])
            mu = opti.fmin_l_bfgs_b(likelihood, x0=mulist[like.argmin()], approx_grad=True, bounds=[b_mu], maxfun=50000)[0]
            return t0, mu, ys

        truth = pelist[pelist['TriggerNo'] == ent[i]['TriggerNo']]
        # t0_t = t0_truth[i]['T0']
        # mu_t = len(truth)
        # 1st FBMP
        time_fbmp_start = time.time()
        factor = np.sqrt(np.diag(np.matmul(A.T, A)).mean())
        # normalize matrix A to unit norm
        A = np.matmul(A, np.diag(1. / np.sqrt(np.diag(np.matmul(A.T, A)))))
        # initalize the expect pe
        la = mu_t * wff.convolve_exp_norm(tlist - t0_t, Tau, Sigma) / n + 1e-8
        xmmse, xmmse_star, psy_star, nu_star, T_star, d_tot_i, d_max_i, expectx, expectx_star = mcp.fbmpr_custom_deactive(wave_r, A, la, spe_pre[cid]['std'] ** 2, (gsigma * factor / gmu) ** 2, factor, int(D * mu_t), einsM = mcp.hnu2PEmatrixLog, stop=5,pmax=500, truth=truth, i=i, left=left_wave, right=right_wave, tlist=tlist, gmu=gmu, para=p)#wff.fbmpr_fxn_reduced(wave_r, A, la, spe_pre[cid]['std'] ** 2, (gsigma * factor / gmu) ** 2, factor, D, stop=2)

        # print('fbmp:{},fbmpmax:{},truth:{},mu_t:{}'.format(np.sum(xmmse),np.sum(xmmse_star[0]),np.sum(pelist['TriggerNo']==i),mu_t))
        
        delta0 += np.sum(xmmse)-np.sum(pelist['TriggerNo']==i)
        delta1 += np.sum(xmmse_star[0])-np.sum(pelist['TriggerNo']==i)
        time_fbmp = time_fbmp + time.time() - time_fbmp_start
        tlist_pan = np.sort(np.unique(np.hstack(np.arange(0, window)[:, None] + np.arange(0, 1, 1 / n))))
        As = np.zeros((len(xmmse_star), len(tlist_pan)),dtype=int)
        As[:, np.isin(tlist_pan, tlist)] = xmmse_star
        logpys_star = nu_star-np.log(poisson.pmf(xmmse_star,la)).sum(axis=1)
        #fitresult = mcp.optimizeBotht0mulogBS(tlist_pan,Tau,Sigma,As,logpys_star,tlist[0])
        fitresult = mcp.optimizeBotht0mulog(tlist_pan,Tau,Sigma,As,logpys_star,t0_t)

        if not fitresult.success:
            print('{}:{}'.format(i, fitresult.message))
        index_star = mcp.findNonzero(As)
        recont0mu[i-a0] = (fitresult.x[0], fitresult.x[1], fitresult.fun, int(fitresult.success),mcp.likelihoodlogBS([t0_truth[i]['T0'],len(truth)], tlist_pan,Tau,Sigma,As,logpys_star,index_star,1/n),t0_truth[i]['T0'],len(truth))
        fbmpPE[i-a0] = (np.sum(xmmse),np.sum(xmmse_star[0]),np.sum(expectx/factor),np.sum(expectx_star[0]/factor), np.sum(pelist['TriggerNo']==i), mu_t)
        pet = np.repeat(tlist[xmmse_star[0] > 0], xmmse_star[0][xmmse_star[0] > 0])
        cha = np.repeat(expectx_star[0][xmmse_star[0] > 0] / factor / xmmse_star[0][xmmse_star[0] > 0], xmmse_star[0][xmmse_star[0] > 0])
        # print(fbmpPE[i-a0])
        # print(recont0mu[i-a0])
        end = start + len(cha)
        dt['HitPosInWindow'][start:end] = pet
        dt['Charge'][start:end] = cha
        dt['TriggerNo'][start:end] = ent[i]['TriggerNo']
        dt['ChannelID'][start:end] = ent[i]['ChannelID']
        start = end
    # print('average:{},max:{}'.format(delta0/100,delta1/100))
    dt = dt[:end]
    dt = np.sort(dt, kind='stable', order=['TriggerNo', 'ChannelID'])
    return dt, recont0mu, fbmpPE, d_tot, d_max, time_fbmp

if args.Ncpu == 1:
    slices = [[0, N]]
else:
    chunk = N // args.Ncpu + 1
    slices = np.vstack((np.arange(0, N, chunk), np.append(np.arange(chunk, N, chunk), N))).T.astype(int).tolist()
# slices = [0, N]

print('Initialization finished, real time {0:.02f}s, cpu time {1:.02f}s'.format(time.time() - global_start, time.process_time() - cpu_global_start))
tic = time.time()
cpu_tic = time.process_time()

ent = np.sort(ent, kind='stable', order=['TriggerNo', 'ChannelID'])
Chnum = len(np.unique(ent['ChannelID']))
e_pel = ent['TriggerNo'] * Chnum + ent['ChannelID']
e_pel, i_pel = np.unique(e_pel, return_index=True)
i_pel = np.append(i_pel, len(ent))

opdt = np.dtype([('TriggerNo', np.uint32), ('ChannelID', np.uint32), ('HitPosInWindow', np.float64), ('Charge', np.float64)])

if method == 'poissonfbmp':
    sdtp = np.dtype([('TriggerNo', np.uint32), ('ChannelID', np.uint32), ('tscharge', np.float64), ('tswave', np.float64), ('mucharge', np.float64), ('muwave', np.float64)])
    ts = np.zeros(N, dtype=sdtp)
    ts['TriggerNo'] = ent['TriggerNo']
    ts['ChannelID'] = ent['ChannelID']
    with Pool(min(args.Ncpu, cpu_count())) as pool:
        result = pool.starmap(partial(fbmp_inference), slices)
    # result = fbmp_inference(slices[0],slices[1])
    # As = result[0]
    # recon = result[1]
    # fbmppe = result[2]
    # d_tot = result[3]
    # d_max = result[4]
    # time_fbmp = result[5].mean()
    As = np.hstack([result[i][0] for i in range(len(slices))])
    recon = np.hstack([result[i][1] for i in range(len(slices))])
    fbmppe = np.hstack([result[i][2] for i in range(len(slices))])
    d_tot = np.hstack([result[i][3] for i in range(len(slices))])
    d_max = np.hstack([result[i][4] for i in range(len(slices))])
    time_fbmp = np.hstack([result[i][5] for i in range(len(slices))]).mean()
    print('FBMP finished, real time {0:.02f}s'.format(time_fbmp))

    As = np.sort(As, kind='stable', order=['TriggerNo', 'ChannelID'])

print('Prediction generated, real time {0:.02f}s, cpu time {1:.02f}s'.format(time.time() - tic, time.process_time() - cpu_tic))

with h5py.File(fopt, 'w') as opt:
    pedset = opt.create_dataset('photoelectron', data=As, compression='gzip')
    pedset.attrs['Method'] = method
    pedset.attrs['mu'] = Mu
    pedset.attrs['tau'] = Tau
    pedset.attrs['sigma'] = Sigma
    opt.create_dataset('recon', data=recon,compression='gzip')
    opt.create_dataset('fbmp', data=fbmppe,compression='gzip')
    print('The output file path is {}'.format(fopt))

print('Finished! Consuming {0:.02f}s in total, cpu time {1:.02f}s.'.format(time.time() - global_start, time.process_time() - cpu_global_start))