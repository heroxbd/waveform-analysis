import os
import time
import argparse
from functools import partial
from multiprocessing import Pool, cpu_count

import h5py
import numpy as np
from numpy.lib import recfunctions
import scipy.optimize as opti
from scipy.interpolate import interp1d
from scipy.stats import poisson
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import wf_func as wff

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', type=str, help='output file')
psr.add_argument('ipt', type=str, help='input file')
psr.add_argument('--Ncpu', dest='Ncpu', type=int, default=100)
psr.add_argument('--ref', type=str, nargs='+', help='reference file')
args = psr.parse_args()

global_start = time.time()
cpu_global_start = time.process_time()

def start_time(a0, a1):
    stime = np.empty(a1 - a0)
    for i in range(a0, a1):
        hitt = photoelectron[i_cha[i]:i_cha[i+1]]['HitPosInWindow'].astype(np.float64)
        char = photoelectron[i_cha[i]:i_cha[i+1]]['Charge']
        t0, _ = wff.likelihoodt0(hitt, char=char, gmu=gmu, Tau=Tau, Sigma=Sigma, mode='charge')
        stime[i - a0] = t0
    return stime

spe_pre = wff.read_model(args.ref[1], wff.nshannon)
p = spe_pre[0]['parameters']
with h5py.File(args.ipt, 'r', libver='latest', swmr=True) as ipt, h5py.File(args.ref[0], 'r', libver='latest', swmr=True) as ref:
    gmu = ref['SimTriggerInfo/PEList'].attrs['gmu']
    gsigma = ref['SimTriggerInfo/PEList'].attrs['gsigma']
    s0 = spe_pre[0]['std'] / np.linalg.norm(spe_pre[0]['spe'])
    method = ipt['photoelectron'].attrs['Method']
    Mu = ipt['photoelectron'].attrs['mu']
    Tau = ipt['photoelectron'].attrs['tau']
    Sigma = ipt['photoelectron'].attrs['sigma']
    photoelectron = ipt['photoelectron'][:]
    window = len(ref['Readout/Waveform'][:][0]['Waveform'][::wff.nshannon])
    n = 1 if min(photoelectron['HitPosInWindow'] % 1) == 0 else min(photoelectron['HitPosInWindow'] % 1)
    tlist_pan = np.sort(np.unique(np.hstack(np.arange(0, window)[:, None] + np.arange(0, 1, 1 / n))))
    Chnum = len(np.unique(photoelectron['ChannelID']))
    photoelectron = np.sort(photoelectron, kind='stable', order=['TriggerNo', 'ChannelID', 'HitPosInWindow'])
    e_cha = photoelectron['TriggerNo'] * Chnum + photoelectron['ChannelID']
    e_cha, i_cha = np.unique(e_cha, return_index=True)
    i_cha = np.append(i_cha, len(photoelectron))
    N = len(e_cha)
    tcdtp = np.dtype([('TriggerNo', np.uint32), ('ChannelID', np.uint32)])
    tc = np.zeros(len(photoelectron), dtype=tcdtp)
    tc['TriggerNo'] = photoelectron['TriggerNo']
    tc['ChannelID'] = photoelectron['ChannelID']
    tc = np.unique(tc)
    if method in ['fbmp', 'mcmc']:
    # if 'starttime' in ipt:
    # if False:
        ts = ipt['starttime'][:]
    else:
        if method == 'takara':
            ts_cpu = ipt['starttime_cpu'][:]
        sdtp = np.dtype([('TriggerNo', np.uint32), ('ChannelID', np.uint32), ('tscharge', np.float64), ('tswave', np.float64), ('mucharge', np.float64), ('muwave', np.float64), ('consumption', np.float64)])
        ts = np.zeros(N, dtype=sdtp)
        ts['TriggerNo'] = tc['TriggerNo']
        ts['ChannelID'] = tc['ChannelID']
        e_ans, i_ans = np.unique(photoelectron['TriggerNo'] * Chnum + photoelectron['ChannelID'], return_index=True)
        i_ans = np.append(i_ans, len(photoelectron))
        cha_sum = np.array([photoelectron[i_ans[i]:i_ans[i+1]]['Charge'].sum() for i in range(len(e_ans))]) / gmu
        ts['muwave'] = cha_sum
        ts['consumption'] = ipt['starttime']['consumption'][:]
        chunk = N // args.Ncpu + 1
        slices = np.vstack((np.arange(0, N, chunk), np.append(np.arange(chunk, N, chunk), N))).T.astype(int).tolist()
        with Pool(min(args.Ncpu, cpu_count())) as pool:
            result = pool.starmap(partial(start_time), slices)
        ts['tscharge'] = np.hstack(result)
        ts['tswave'] = np.full(N, np.nan)
        ts['mucharge'] = np.full(N, np.nan)
    if method == 'fbmp':
        mu = ipt['starttime'].attrs['mu']
        sigmamu = ipt['starttime'].attrs['sigmamu']
    else:
        N_tot = round(N / (1 - poisson.cdf(0, Mu)))
        charge = photoelectron['Charge'] / gmu
        N_add = N_tot - N
        dkl = lambda mu: -np.log(mu) * charge.sum() + mu * N_tot
        # mu, fval, _ = opti.fmin_l_bfgs_b(dkl, x0=[Mu * 1.05], approx_grad=True, bounds=[[mulist[0], mulist[-1]]], maxfun=500000)
        mulist = np.arange(max(1e-8, Mu - 2 * np.sqrt(Mu)), Mu + 2 * np.sqrt(Mu), 0.1)
        mu = charge.sum() / N_tot
        fval = dkl(mu)
        sigmamu_est = np.sqrt(mu / N_tot)
        mulist = np.sort(np.append(np.arange(max(1e-8, mu - 3 * sigmamu_est), mu + 3 * sigmamu_est, sigmamu_est / 50), mu))
        dkl_mu = np.vectorize(dkl)(mulist)
        # mu_func = interp1d(mulist, dkl_mu, bounds_error=False, fill_value='extrapolate')
        # logLvdelta = np.vectorize(lambda mu_t: np.abs(mu_func(mu_t) - fval - 0.5))
        # sigmamu = abs(opti.fmin_l_bfgs_b(logLvdelta, x0=[mulist[np.abs(dkl_mu - fval - 0.5).argmin()]], approx_grad=True, bounds=[[mulist[0], mulist[-1]]], maxfun=500000)[0] - mu) * np.sqrt(N_tot)

        fig = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(1, 1, figure=fig, left=0.15, right=0.95, top=0.95, bottom=0.1, wspace=0.3, hspace=0.3)
        # fig.tight_layout()
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(mulist, dkl_mu - fval, label=r'$D_{KL}$')
        ax.axvline(x=mu, color='r')
        ax.axhline(y=0.5, color='k', linestyle='dashed', alpha=0.5)
        ax.axhline(y=0, color='k')
        ax.grid()
        ax.set_xlabel('mu')
        ax.set_ylim(-0.1, 2)
        ax.set_ylabel(r'$D_{KL}$')
        ax.legend(loc='upper right')
        fig.savefig(os.path.splitext(args.opt)[0] + '.png')

        sigmamu = np.sqrt(charge.sum() / N_tot)
print('mu is {0:.3f}, sigma_mu is {1:.3f}'.format(mu.item(), sigmamu.item()))

with h5py.File(args.opt, 'w') as opt:
    dset = opt.create_dataset('starttime', data=ts, compression='gzip')
    dset.attrs['Method'] = method
    dset.attrs['mu'] = Mu
    dset.attrs['tau'] = Tau
    dset.attrs['sigma'] = Sigma
    dset.attrs['mu'] = mu
    dset.attrs['sigmamu'] = sigmamu
    if method == 'takara':
        dset = opt.create_dataset('starttime_cpu', data=ts_cpu, compression='gzip')
    print('The output file path is {}'.format(args.opt))

print('Finished! Consuming {0:.02f}s in total, cpu time {1:.02f}s.'.format(time.time() - global_start, time.process_time() - cpu_global_start))
