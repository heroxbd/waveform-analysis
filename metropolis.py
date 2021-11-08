import os
import sys
import re
import time
import math
import argparse
import pickle
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
from tqdm import tqdm

import wf_func as wff

global_start = time.time()
cpu_global_start = time.process_time()

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', type=str, help='output file')
psr.add_argument('ipt', type=str, help='input file')
psr.add_argument('--ref', type=str, help='reference file')
args = psr.parse_args()

fipt = args.ipt
fopt = args.opt
reference = args.ref

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
    PEList = ipt['SimTriggerInfo/PEList'][:]

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

n = 2
b_t0 = [0., 600.]

print('Initialization finished, real time {0:.02f}s, cpu time {1:.02f}s'.format(time.time() - global_start, time.process_time() - cpu_global_start))
tic = time.time()
cpu_tic = time.process_time()

ent = np.sort(ent, kind='stable', order=['TriggerNo', 'ChannelID'])
Chnum = len(np.unique(ent['ChannelID']))

dt = [('TriggerNo', np.uint32), ('ChannelID', np.uint32), ('istar', np.uint16), ('flip', np.int8), ('delta_nu', np.float64)]
mu0_dt = [('TriggerNo', np.uint32), ('ChannelID', np.uint32), ('mu_t', np.float64)]
TRIALS = 10000

with h5py.File(fopt, 'w') as opt:
    sample = opt.create_dataset('sample', shape=(N*TRIALS,), dtype=dt)
    mu0 = opt.create_dataset('mu0', shape=(N,), dtype=mu0_dt)
    
    print('The output file path is {}'.format(fopt))

    for ie, e in enumerate(ent):
        time_fbmp_start = time.time()
        eid = e["TriggerNo"]
        cid = e['ChannelID']
        assert cid == 0
        PEs = PEList[np.logical_and(PEList["TriggerNo"] == eid, PEList["PMTId"] == cid)]
        NPE_t = len(PEs)

        wave = e['Waveform'].astype(np.float64) * spe_pre[cid]['epulse']

        # initialization
        A, y, tlist, t0_t, t0_delta, cha, left_wave, right_wave = wff.initial_params(wave[::wff.nshannon], spe_pre[e['ChannelID']], Tau, Sigma, gmu, Thres['lucyddm'], p, is_t0=True, is_delta=False, n=n, nshannon=1)
        cha += 1e-8 # for completeness of the random walk.
        t0_t = t0_truth['T0'][ie] # override with truth to debug mu
        mu_t = abs(y.sum() / gmu)
        mu0[ie] = (eid, cid, mu_t)
        # Eq. (9) where the columns of A are taken to be unit-norm.
        mus = np.sqrt(np.diag(np.matmul(A.T, A)))
        A = A / mus

        '''
        p1: prior probability for each bin.
        sig2w: variance of white noise.
        sig2s: variance of signal x_i.
        mus: mean of signal x_i.
        TRIALS: number of Metropolis steps.
        '''
        
        p1 = mu_t * wff.convolve_exp_norm(tlist - t0_t, Tau, Sigma) / n

        sig2w = spe_pre[cid]['std'] ** 2
        sig2s = (gsigma * mus / gmu) ** 2

        # Only for multi-gaussian with arithmetic sequence of mu and sigma
        # N: number of t bins
        # M: length of the waveform clip
        M, N = A.shape

        # nu: nu for all s_n=0.
        ν = -0.5 * np.linalg.norm(y) ** 2 / sig2w - 0.5 * M * np.log(2 * np.pi)
        ν -= 0.5 * M * np.log(sig2w)
        ν += poisson.logpmf(0, p1).sum()

        # Eq. (29)
        cx = A / sig2w
        # mu = 0 => (y - A * mu -> z)
        z = y
        # model selection vector
        s = np.zeros(N)

        # Metropolis
        istar = np.random.choice(range(N), TRIALS, p=cha/np.sum(cha))
        # -1 PE, +1 PE, 左移 PE, 右移 PE
        flip = np.random.choice((-1, 1, -2, 2), TRIALS)
        Δν_history = np.zeros(TRIALS) # list of Δν's
        for i, accept in enumerate(np.log(np.random.uniform(size=TRIALS))):
            t = istar[i] # the time bin

            if s[t] == 0: # 下边界
                flip[i] = 1 # 不论是 -1 还是 +-2，都转换成 +1
                # Q(1->0) / Q(0->1) = 1 / 4
                # 从 0 开始只有一种跳跃可能到 1，因此需要惩罚
                accept += np.log(4)
            elif s[t] == 1 and flip[i] == -1:
                # 1 -> 0: 行动后从 0 脱出的几率大，需要鼓励
                accept -= np.log(4)

            if abs(flip[i]) == 2:
                if t == 0: # 左边界
                    if flip[i] == -2: # 在最左边，不能有 -2 向左移动
                        flip[i] = 2
                    if flip[i] == 2:
                        # Q(移 1->0) / Q(移 0->1) = 1 / 2
                        # 从 0 移动只能到 1，因此需要惩罚
                        accept += np.log(2)
                elif t == 1 and flip[i] == -2:
                    accept -= np.log(2)

                if t == N-1: # 右边界
                    if flip[i] == 2: # 在最右边，不能有 2 向右移动
                        flip[i] = -2
                    if flip[i] == -2:
                        # 从 N-1 只能到 N-2，惩罚
                        accept += np.log(2)
                elif t == N-2 and flip[i] == 2:
                    accept -= np.log(2)

                t_next = t + 1 if flip[i] == 2 else t - 1
                # Q(移 t_next -> t) / Q(移 t -> t_next)
                # 若 p(t_next) 很大，则应鼓励
                accept -= np.log(cha[t_next] / cha[t])

            def move(cx, z, t, step):
                '''
                step
                ====
                1: 在 t 加一个 PE
                -1: 在 t 减一个 PE
                '''
                fsig2s = step * sig2s[t]
                # Eq. (30) sig2s = 1 sigma^2 - 0 sigma^2
                beta_under = (1 + fsig2s * np.dot(A[:, t], cx[:, t]))
                beta = fsig2s / beta_under

                # Eq. (31) # sign of mus[t] and sig2s[t] cancels
                Δν = 0.5 * (beta * (z @ cx[:, t] + mus[t] / sig2s[t]) ** 2 - mus[t] ** 2 / fsig2s)
                # sign of space factor in Eq. (31) is reversed.  Because Eq. (82) is in the denominator.
                Δν -= 0.5 * np.log(beta_under) # space
                # poisson
                Δν += step * np.log(p1[t])
                if step == 1:
                    Δν -= np.log(s[t] + 1)
                else: # step == -1
                    Δν += np.log(s[t])

                # accept, prepare for the next
                # Eq. (33) istar is now n_pre.  It crosses n_pre and n, thus is in vector form.
                Δcx = -np.einsum('n,m,mp->np', beta * cx[:, t], cx[:, t], A, optimize=True)
                # Eq. (34)
                Δz = -step * A[:, t] * mus[t]
                return Δν, Δcx, Δz

            if abs(flip[i]) == 2:
                Δν0, Δcx0, Δz0 = move(cx, z, t, -1)
                Δν1, Δcx1, Δz1 = move(cx + Δcx0, z + Δz0, t_next, 1)
                Δν = Δν0 + Δν1
                Δcx = Δcx0 + Δcx1
                Δz = Δz0 + Δz1
            elif abs(flip[i]) == 1:
                Δν, Δcx, Δz = move(cx, z, t, flip[i])

            if Δν >= accept:
                cx += Δcx
                z += Δz
                Δν_history[i] = Δν

                if abs(flip[i]) == 2:
                    s[t] -= 1
                    s[t_next] += 1
                elif abs(flip[i]) == 1:
                    s[t] += flip[i] # update state
            else:
                # reject proposal
                flip[i] = 0
                Δν_history[i] = 0

        sample[ie*TRIALS:(ie+1)*TRIALS] = list(zip(np.repeat(eid, TRIALS), 
                                                   np.repeat(cid, TRIALS), 
                                                   istar, flip, Δν_history))

print('Prediction generated, real time {0:.02f}s, cpu time {1:.02f}s'.format(time.time() - tic, time.process_time() - cpu_tic))

print('Finished! Consuming {0:.02f}s in total, cpu time {1:.02f}s.'.format(time.time() - global_start, time.process_time() - cpu_global_start))
