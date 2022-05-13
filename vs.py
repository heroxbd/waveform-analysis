import os
import sys
import re
import time
import math
import csv
import argparse
import itertools
import warnings
warnings.filterwarnings('ignore')

import h5py
import numpy as np
from scipy import stats
from scipy.stats import norm, poisson, uniform, chi2, t, expon
from scipy.integrate import quad
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec

import wf_func as wff

import matplotlib
matplotlib.use('pgf')

plt.rcParams['font.size'] = 14

psr = argparse.ArgumentParser()
psr.add_argument('--conf', type=str, help='configuration of tau & sigma')
args = psr.parse_args()

std = wff.std

with open(args.conf) as f:
    f_csv = csv.reader(f, delimiter=' ')
    Tau = next(f_csv)
    Tau = [float(i) for i in Tau]
    Sigma = next(f_csv)
    Sigma = [float(i) for i in Sigma]

default_method = 'fsmp'

filelist = os.listdir(f'result/{default_method}/solu')
filelist = [f for f in filelist if f[0] != '.' and os.path.splitext(f)[-1] == '.h5']
numbers = [[float(i) for i in f[:-3].split('-')] for f in filelist]
stype = np.dtype([('mu', np.float64), ('tau', np.float64), ('sigma', np.float64), ('n', np.uint), ('std1sttruth', np.float64), ('stdtruth', np.float64), ('std', np.float64), ('stdone', np.float64), ('bias1sttruth', np.float64), ('biastruth', np.float64), ('bias', np.float64), ('biasone', np.float64), ('wdist', np.float64, 3), ('RSS', np.float64, 3), ('N', np.uint), ('consumption', np.float64, 3), ('stdsuccess', np.uint), ('stdonesuccess', np.uint)])
mtsi = np.zeros(len(numbers), dtype=stype)
mtsi['mu'] = np.array([i[0] for i in numbers])
mtsi['tau'] = np.array([i[1] for i in numbers])
mtsi['sigma'] = np.array([i[2] for i in numbers])
mtsi['n'] = np.arange(len(numbers))
mtsi['std1sttruth'] = np.nan
mtsi['stdtruth'] = np.nan
mtsi['std'] = np.nan
mtsi['stdone'] = np.nan
mtsi['bias1sttruth'] = np.nan
mtsi['biastruth'] = np.nan
mtsi['bias'] = np.nan
mtsi['biasone'] = np.nan
mtsi['wdist'] = np.nan
mtsi['RSS'] = np.nan
mtsi['consumption'] = np.nan
mtsi = np.sort(mtsi, kind='stable', order=['mu', 'tau', 'sigma'])

mts = {'firstthres':mtsi.copy(), 'threshold':mtsi.copy(), 'findpeak':mtsi.copy(), 'fftrans':mtsi.copy(), 'lucyddm':mtsi.copy(), 'takara':mtsi.copy(), 'xiaopeip':mtsi.copy(), 'mcmc':mtsi.copy(), 'fsmp':mtsi.copy()}
deltalabel = {'1st':'\mathrm{1st}', 'tru':'\mathrm{ALL}', 'findpeak':'\mathrm{FindPeak}', 'threshold':'\mathrm{Shift}', 'fftrans':'\mathrm{FFT}', 'lucyddm':'\mathrm{LucyDDM}', 'xiaopeip':'\mathrm{Fitting}', 'takara':'\mathrm{CNN}', 'fsmp':'\mathrm{FSMP}', 'fsmpone':'\mathrm{FSMPmax}', 'mcmc':'\mathrm{MCMC}', 'mcmcone':'\mathrm{MCMCcha}', 'firstthres':'\mathrm{1stthres}'}
label = {'1st':'\mathrm{1st}', 'tru':'\mathrm{ALL}', 'findpeak':'\mathrm{Peak\ finding}', 'threshold':'\mathrm{Waveform\ shifting}', 'fftrans':'\mathrm{Fourier\ deconvolution}', 'lucyddm':'\mathrm{LucyDDM}', 'xiaopeip':'\mathrm{Fitting}', 'takara':'\mathrm{CNN}', 'fsmp':'\mathrm{FSMP}', 'fsmpone':'\mathrm{FSMPmax}', 'mcmc':'\mathrm{MCMC}', 'firstthres':'\mathrm{1stthres}'}
marker = {'1st':'o', 'tru':'h', 'findpeak':',', 'threshold':'1', 'fftrans':'+', 'lucyddm':'p', 'xiaopeip':'*', 'takara':'x', 'fsmp':'s', 'fsmpone':'^', 'mcmc':'X', 'mcmcone':'>', 'firstthres':'>'}
color = {'1st':'g', 'tru':'k', 'findpeak':'C1', 'threshold':'C2', 'fftrans':'m', 'lucyddm':'y', 'xiaopeip':'c', 'takara':'C0', 'fsmp':'r', 'fsmpone':'b', 'mcmc':'C4', 'mcmcone':'C5', 'firstthres':'C6'}
jit = 0.05
jitter = {'mcmcone':-5 * jit, 'mcmc':-4 * jit, 'tru':-3 * jit, 'fsmp':-2 * jit, 'fsmpone':-1 * jit, 'lucyddm':0 * jit, 'xiaopeip':1 * jit, 'takara':2 * jit, '1st':3 * jit, 'fftrans':4 * jit, 'findpeak':5 * jit, 'threshold':6 * jit, 'firstthres':7 * jit}

alpha = 0.05
for key in tqdm(mts.keys()):
    for i in range(len(mts[key])):
        f = filelist[mts[key][i]['n']]
        mu = mts[key][i]['mu']
        tau = mts[key][i]['tau']
        sigma = mts[key][i]['sigma']
        waveform = 'waveform'
        try:
            with h5py.File(os.path.join('result', key, 'solu', f), 'r', libver='latest', swmr=True) as soluf, h5py.File(os.path.join('result', key, 'dist', f), 'r', libver='latest', swmr=True) as distf, h5py.File(os.path.join(waveform, f), 'r', libver='latest', swmr=True) as wavef:
                time = soluf['starttime'][:]
                if key != 'fsmp':
                    mu_hat = time['muwave']
                else:
                    mu_hat = np.inf
                record = distf['Record'][:]
                start = wavef['SimTruth/T'][:]
                gmu = wavef['SimTriggerInfo/PEList'].attrs['gmu']
                Npe = wavef['Readout/Waveform']['Npe']
                gsigma = wavef['SimTriggerInfo/PEList'].attrs['gsigma']
                # r = wavef['SimTruth/T'].attrs['r']
                r = np.inf
            mts[key][i]['N'] = len(start)
            vali = np.abs(time['tscharge'] - start['T0'] - np.mean(time['tscharge'] - start['T0'])) <= r * np.std(time['tscharge'] - start['T0'], ddof=-1)
            mts[key][i]['std1sttruth'] = np.std(start['ts1sttruth'] - start['T0'], ddof=-1)
            mts[key][i]['stdtruth'] = np.std(start['tstruth'] - start['T0'], ddof=-1)
            mts[key][i]['std'], mts[key][i]['stdsuccess'] = wff.stdrmoutlier(time['tscharge'] - start['T0'], r)
            mts[key][i]['bias1sttruth'] = np.mean(start['ts1sttruth'] - start['T0'])
            mts[key][i]['biastruth'] = np.mean(start['tstruth'] - start['T0'])
            mts[key][i]['bias'] = np.mean(time['tscharge'][vali] - start['T0'][vali])
            # mts[key][i]['std1sttruth'] = np.std(start['ts1sttruth'] - start['T0'] - tau / Npe, ddof=-1)
            # mts[key][i]['stdtruth'] = np.std(start['tstruth'] - start['T0'] - tau / Npe, ddof=-1)
            # mts[key][i]['std'], mts[key][i]['stdsuccess'] = wff.stdrmoutlier(time['tscharge'] - start['T0'] - tau / mu_hat, r)
            # mts[key][i]['bias1sttruth'] = np.mean(start['ts1sttruth'] - start['T0'] - tau / Npe)
            # mts[key][i]['biastruth'] = np.mean(start['tstruth'] - start['T0'] - tau / Npe)
            # mts[key][i]['bias'] = np.mean(time['tscharge'][vali] - start['T0'][vali] - tau / mu_hat)
            mts[key][i]['wdist'] = np.insert(np.percentile(record['wdist'][vali], [alpha * 100, 100 - alpha * 100]), 1, record['wdist'][vali].mean())
            mts[key][i]['RSS'] = np.insert(np.percentile(record['RSS'][vali], [alpha * 100, 100 - alpha * 100]), 1, record['RSS'][vali].mean())
            if not np.any(np.isnan(time['tswave'][vali])):
                mts[key][i]['stdone'], mts[key][i]['stdonesuccess'] = wff.stdrmoutlier(time['tscharge'] - start['T0'], r)
                mts[key][i]['biasone'] = np.mean(time['tscharge'][vali] - start['T0'][vali])
                vali = np.abs(time['tswave'] - start['T0'] - np.mean(time['tswave'] - start['T0'])) < r * np.std(time['tswave'] - start['T0'], ddof=-1)
                mts[key][i]['std'], mts[key][i]['stdsuccess'] = wff.stdrmoutlier(time['tswave'] - start['T0'], r)
                mts[key][i]['bias'] = np.mean(time['tswave'][vali] - start['T0'][vali])
            try:
                mts[key][i]['consumption'] = np.insert(np.percentile(time['consumption'][vali], [alpha * 100, 100 - alpha * 100]), 1, time['consumption'][vali].mean())
            except:
                pass
        except:
            pass

dhigh = np.array([[np.max(mts[key]['std1sttruth']), np.max(mts[key]['stdtruth']), np.max(mts[key]['std']), np.max(mts[key]['stdone'])] for key in mts.keys()])
dhigh = np.max(dhigh[~np.isnan(dhigh)]) * 1.05
whigh = np.array([[np.max(mts[key]['wdist'])] for key in mts.keys()])
whigh = np.max(whigh[~np.isnan(whigh)]) * 1.05
rhigh = np.array([[np.max(mts[key]['RSS'])] for key in mts.keys()])
rhigh = np.max(rhigh[~np.isnan(rhigh)]) * 1.05

lim = {'deltadiv':np.array([[0.3, 0.5]]), 'wdist':np.array([[1.5, 2.0]]), 'rss':np.array([[150, 300]])}

fig_t0_resolution = plt.figure(figsize=(len(Tau) * 5, len(Sigma) * 3))
gs_t0_resolution = gridspec.GridSpec(len(Sigma), len(Tau), figure=fig_t0_resolution, left=0.1, right=0.8, top=0.92, bottom=0.15, wspace=0.3, hspace=0.35)
fig_t0_bias = plt.figure(figsize=(len(Tau) * 5, len(Sigma) * 3))
gs_t0_bias = gridspec.GridSpec(len(Sigma), len(Tau), figure=fig_t0_bias, left=0.1, right=0.8, top=0.92, bottom=0.15, wspace=0.3, hspace=0.35)
fig_t0_resolution_ratio = plt.figure(figsize=(len(Tau) * 5, len(Sigma) * 3))
gs_t0_resolution_ratio = gridspec.GridSpec(len(Sigma), len(Tau), figure=fig_t0_resolution_ratio, left=0.1, right=0.8, top=0.92, bottom=0.15, wspace=0.3, hspace=0.35)
fig_wdist = plt.figure(figsize=(len(Tau) * 5, len(Sigma) * 3))
gs_wdist = gridspec.GridSpec(len(Sigma), len(Tau), figure=fig_wdist, left=0.1, right=0.8, top=0.92, bottom=0.15, wspace=0.3, hspace=0.35)
fig_rss = plt.figure(figsize=(len(Tau) * 5, len(Sigma) * 3))
gs_rss = gridspec.GridSpec(len(Sigma), len(Tau), figure=fig_rss, left=0.1, right=0.8, top=0.92, bottom=0.15, wspace=0.3, hspace=0.35)
keylist = list(mts.keys())
badkey = ['findpeak', 'threshold', 'fftrans', 'mcmc', 'firstthres']
for i, sigma in enumerate(Sigma):
    for j, tau in enumerate(Tau):
        ax = fig_t0_resolution.add_subplot(gs_t0_resolution[i, j])
        stdlist = mts[default_method][(mts[default_method]['tau'] == tau) & (mts[default_method]['sigma'] == sigma)]
        yerr1st = np.vstack([stdlist['std1sttruth']-np.sqrt(np.power(stdlist['std1sttruth'],2)*stdlist['N']/chi2.ppf(1-alpha, stdlist['N'])), np.sqrt(np.power(stdlist['std1sttruth'],2)*stdlist['N']/chi2.ppf(alpha, stdlist['N']))-stdlist['std1sttruth']])
        yerrall = np.vstack([stdlist['stdtruth']-np.sqrt(np.power(stdlist['stdtruth'],2)*stdlist['N']/chi2.ppf(1-alpha, stdlist['N'])), 
                             np.sqrt(np.power(stdlist['stdtruth'],2)*stdlist['N']/chi2.ppf(alpha, stdlist['N']))-stdlist['stdtruth']])
        ax.errorbar(stdlist['mu'] + jitter['1st'], stdlist['std1sttruth'], yerr=yerr1st, c=color['1st'], label='$'+deltalabel['1st']+'$', marker=marker['1st'])
        ax.errorbar(stdlist['mu'] + jitter['tru'], stdlist['stdtruth'], yerr=yerrall, c=color['tru'], label='$'+deltalabel['tru']+'$', marker=marker['tru'])
        for m in ['mcmc', 'fsmp']:
            stdlistwav = mts[m][(mts[m]['tau'] == tau) & (mts[m]['sigma'] == sigma)]
            yerrwav = np.vstack([stdlistwav['stdone']-np.sqrt(np.power(stdlistwav['stdone'],2)*stdlistwav['N']/chi2.ppf(1-alpha, stdlistwav['N'])), np.sqrt(np.power(stdlistwav['stdone'],2)*stdlistwav['N']/chi2.ppf(alpha, stdlistwav['N']))-stdlistwav['stdone']])
            ax.errorbar(stdlistwav['mu'] + jitter[m + 'one'], stdlistwav['stdone'], yerr=yerrwav, c=color[m + 'one'], label='$'+deltalabel[m + 'one']+'$', marker=marker[m + 'one'])
        for key in keylist:
            stdlist = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
            yerrcha = np.vstack([stdlist['std']-np.sqrt(np.power(stdlist['std'],2)*stdlist['N']/chi2.ppf(1-alpha, stdlist['N'])), np.sqrt(np.power(stdlist['std'],2)*stdlist['N']/chi2.ppf(alpha, stdlist['N']))-stdlist['std']])
            ax.errorbar(stdlist['mu'] + jitter[key], stdlist['std'], yerr=yerrcha, c=color[key], label='$'+deltalabel[key]+'$', marker=marker[key])
        ax.set_xlabel(r'$\mu$')
        ax.set_ylabel(r'$\sigma/\si{ns}$')
        ax.set_title(fr'$\tau_l={tau}\si{{ns}},\,\sigma_l={sigma}\si{{ns}}$')
        # ax.set_yscale('log')
        ax.grid()
        if i == len(Sigma) - 1 and j == len(Tau) - 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1., 0.9), prop={'size': 5})

        #
        # bias of t_0 estimation
        # 
        ax = fig_t0_bias.add_subplot(gs_t0_bias[i, j])
        stdlist = mts[default_method][(mts[default_method]['tau'] == tau) & (mts[default_method]['sigma'] == sigma)]
        # yerr1st = np.vstack([-t.ppf(alpha, stdlist['N'])*stdlist['std1sttruth']/np.sqrt(stdlist['N']), t.ppf(1-alpha, stdlist['N'])*stdlist['std1sttruth']/np.sqrt(stdlist['N'])])
        yerrall = np.vstack([-t.ppf(alpha, stdlist['N'])*stdlist['stdtruth']/np.sqrt(stdlist['N']), 
                             t.ppf(1-alpha, stdlist['N'])*stdlist['stdtruth']/np.sqrt(stdlist['N'])])
        # ax.errorbar(stdlist['mu'] + jitter['1st'], stdlist['bias1sttruth'], yerr=yerr1st, c=color['1st'], label='$'+deltalabel['1st']+'$', marker=marker['1st'])
        ax.errorbar(stdlist['mu'] + jitter['tru'], stdlist['biastruth'], yerr=yerrall, c=color['tru'], label='$'+deltalabel['tru']+'$', marker=marker['tru'])
        for key in keylist:
            if key in badkey:
                continue
            stdlist = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
            yerrcha = np.vstack([-t.ppf(alpha, stdlist['N'])*stdlist['std']/np.sqrt(stdlist['N']), t.ppf(1-alpha, stdlist['N'])*stdlist['std']/np.sqrt(stdlist['N'])])
            ax.errorbar(stdlist['mu'] + jitter[key], stdlist['bias'], yerr=yerrcha, c=color[key], label='$'+deltalabel[key]+'$', marker=marker[key])
        ax.set_ylim(-0.5, 12)
        ax.set_xlabel(r'$\mu$')
        ax.set_ylabel(r'$\mathrm{bias}/\si{ns}$')
        ax.set_title(fr'$\tau_l={tau}\si{{ns}},\,\sigma_l={sigma}\si{{ns}}$')
        # ax.set_yscale('log')
        ax.grid()
        if i == len(Sigma) - 1 and j == len(Tau) - 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1., 0.9))

        #
        # Ratio of sigma^2_{t_0} and sigma^2_{ALL}
        # 
        ax = fig_t0_resolution_ratio.add_subplot(gs_t0_resolution_ratio[i, j])
        for key in keylist:
            if key in badkey:
                continue
            stdlist = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
            # if key == 'fsmp':
            #     yerr = np.vstack([stdlist['stdone'] / stdlist['stdtruth']*(1-1/np.sqrt(stats.f.ppf(1-alpha, stdlist['N']-1, stdlist['N']-1))), (1/np.sqrt(stats.f.ppf(alpha, stdlist['N']-1, stdlist['N']-1))-1)*stdlist['stdone']/stdlist['stdtruth']])
            #     ax.errorbar(stdlist['mu'], stdlist['stdone'] / stdlist['stdtruth'], yerr=yerr, label='$'+deltalabel['fsmpone']+'$', c=color['fsmpone'], marker=marker['fsmpone'])
            yerr = np.vstack([stdlist['std'] / stdlist['stdtruth']*(1-1/np.sqrt(stats.f.ppf(1-alpha, stdlist['N']-1, stdlist['N']-1))), 
                              (1/np.sqrt(stats.f.ppf(alpha, stdlist['N']-1, stdlist['N']-1))-1)*stdlist['std']/stdlist['stdtruth']])
            ax.errorbar(stdlist['mu'] + jitter[key], stdlist['std'] / stdlist['stdtruth'], yerr=yerr, label='$'+deltalabel[key]+'$', c=color[key], marker=marker[key])
        temp_ylim = ax.get_ylim()
        stdlist = mts[default_method][(mts[default_method]['tau'] == tau) & (mts[default_method]['sigma'] == sigma)]
        yerr = np.vstack([stdlist['std'] / stdlist['stdtruth']*(1-1/np.sqrt(stats.f.ppf(1-alpha, stdlist['N']-1, stdlist['N']-1))), (1/np.sqrt(stats.f.ppf(alpha, stdlist['N']-1, stdlist['N']-1))-1)*stdlist['std']/stdlist['stdtruth']])
        ax.errorbar(stdlist['mu'] + jitter['1st'], stdlist['std1sttruth'] / stdlist['stdtruth'], yerr=yerr, label='$'+deltalabel['1st']+'$', c=color['1st'], marker=marker['1st'])
        ax.set_ylim(temp_ylim)
        ax.set_xlabel(r'$\mu$')
        ax.set_ylabel(r'$\mathrm{ratio}$')
        ax.set_title(fr'$\tau_l={tau}\si{{ns}},\,\sigma_l={sigma}\si{{ns}}$')
        # ax.set_ylim(lim['deltadiv'][i, j], 1.01)
        ax.grid()
        if i == len(Sigma) - 1 and j == len(Tau) - 1:
            handles, labels = ax.get_legend_handles_labels()
            # sort both labels and handles by labels
            labels = labels[-1:] + labels[:-1]
            handles = handles[-1:] + handles[:-1]
            ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1., 0.9))

        ax = fig_wdist.add_subplot(gs_wdist[i, j])
        for key in keylist:
            if key in badkey:
                continue
            wdistlist = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
            # ax.plot(wdistlist['mu'] + jitter[key], wdistlist['wdist'], c=color[key], label='$'+label[key]+'$', marker=marker[key])
            yerr = np.vstack([wdistlist['wdist'][:, 1] - wdistlist['wdist'][:, 0], wdistlist['wdist'][:, 2] - wdistlist['wdist'][:, 1]])
            ax.errorbar(wdistlist['mu'] + jitter[key], wdistlist['wdist'][:, 1], yerr=yerr, label='$'+label[key]+'$', c=color[key], marker=marker[key])
            ax.fill_between(wdistlist['mu'] + jitter[key], wdistlist['wdist'][:, 0], wdistlist['wdist'][:, 2], fc=color[key], alpha=0.1, color=None)
        ax.set_xlabel(r'$\mu$')
        ax.set_ylabel(r'$\mathrm{Wasserstein\ Distance}/\si{ns}$')
        ax.set_title(fr'$\tau_l={tau}\si{{ns}},\,\sigma_l={sigma}\si{{ns}}$')
        ax.set_ylim(0, lim['wdist'][i, j])
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.grid()
        if i == len(Sigma) - 1 and j == len(Tau) - 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1., 0.9))

        ax = fig_rss.add_subplot(gs_rss[i, j])
        for key in keylist:
            if key in badkey:
                continue
            rsslist = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
            # ax.plot(rsslist['mu'] + jitter[key], rsslist['RSS'], c=color[key], label='$'+label[key]+'$', marker=marker[key])
            yerr = np.vstack([rsslist['RSS'][:, 1] - rsslist['RSS'][:, 0], rsslist['RSS'][:, 2] - rsslist['RSS'][:, 1]])
            ax.errorbar(rsslist['mu'] + jitter[key], rsslist['RSS'][:, 1], yerr=yerr, label='$'+label[key]+'$', c=color[key], marker=marker[key])
            ax.fill_between(rsslist['mu'] + jitter[key], rsslist['RSS'][:, 0], rsslist['RSS'][:, 2], fc=color[key], alpha=0.1, color=None)
        ax.set_xlabel(r'$\mu$')
        ax.set_ylabel(r'$\mathrm{RSS}/\si{mV}^{2}$')
        ax.set_title(fr'$\tau_l={tau}\si{{ns}},\,\sigma_l={sigma}\si{{ns}}$')
        ax.set_ylim(0, lim['rss'][i, j])
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
        ax.grid()
        if i == len(Sigma) - 1 and j == len(Tau) - 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1., 0.9))
fig_t0_resolution.savefig('Note/figures/vs-sigmat0.pgf')
fig_t0_resolution.savefig('Note/figures/vs-sigmat0.pdf')
fig_t0_resolution.savefig('Note/figures/vs-sigmat0.png')
plt.close(fig_t0_resolution)
fig_t0_resolution_ratio.savefig('Note/figures/vs-sigmat0-r.pgf')
fig_t0_resolution_ratio.savefig('Note/figures/vs-sigmat0-r.pdf')
fig_t0_resolution_ratio.savefig('Note/figures/vs-sigmat0-r.png')
plt.close(fig_t0_resolution_ratio)
fig_t0_bias.savefig('Note/figures/vs-biast0.pgf')
fig_t0_bias.savefig('Note/figures/vs-biast0.pdf')
fig_t0_bias.savefig('Note/figures/vs-biast0.png')
plt.close(fig_t0_bias)
fig_wdist.savefig('Note/figures/vs-wdist.pgf')
fig_wdist.savefig('Note/figures/vs-wdist.pdf')
fig_wdist.savefig('Note/figures/vs-wdist.png')
plt.close(fig_wdist)
fig_rss.savefig('Note/figures/vs-rss.pgf')
fig_rss.savefig('Note/figures/vs-rss.pdf')
fig_rss.savefig('Note/figures/vs-rss.png')
plt.close(fig_rss)

thresfirst = False
marker2 = [['s', '^']]
colors2 = [['r', 'b']]
fig = plt.figure(figsize=(5, 4))
gs = gridspec.GridSpec(1, 1, figure=fig, left=0.15, right=0.92, top=0.92, bottom=0.15, wspace=0.3, hspace=0.2)
ax = fig.add_subplot(gs[0, 0])
stdlist = mts['firstthres'][(mts['firstthres']['tau'] == Tau[0]) & (mts['firstthres']['sigma'] == Sigma[0])]
std1sttruth = np.empty(len(stdlist['mu']))
sigma = 0
tau = max(Tau)
np.random.seed(0)
for i, mu in enumerate(stdlist['mu']):
    N = stdlist['N'][i]
    npe = poisson.ppf(1 - uniform.rvs(scale=1-poisson.cdf(0, mu), size=N), mu).astype(int)
    t0 = np.random.uniform(100., 500., size=N)
    sams = [wff.time(npe[j], tau, sigma) + t0[j] for j in range(N)]
    ts1sttruth = np.array([np.min(sams[j]) for j in range(N)])
    std1sttruth[i] = np.std(ts1sttruth - t0, ddof=-1)
yerr = np.vstack([std1sttruth-np.sqrt(np.power(std1sttruth,2)*stdlist['N']/chi2.ppf(1-alpha, stdlist['N'])), np.sqrt(np.power(std1sttruth,2)*stdlist['N']/chi2.ppf(alpha, stdlist['N']))-std1sttruth])
ax.errorbar(stdlist['mu'], std1sttruth, yerr=yerr, label='$\sigma_'+deltalabel['tru']+fr',\,\tau_l={tau}\si{{ns}},\,\sigma_l={sigma}\si{{ns}}$', marker='o', color='g')
ax.errorbar(stdlist['mu'], std1sttruth, yerr=yerr, label='$\sigma_'+deltalabel['1st']+fr',\,\tau_l={tau}\si{{ns}},\,\sigma_l={sigma}\si{{ns}}$', marker='o', color='g', linestyle='dashed')
for i, sigma in enumerate(Sigma):
    for j, tau in enumerate(Tau):
        stdlist = mts['firstthres'][(mts['firstthres']['tau'] == tau) & (mts['firstthres']['sigma'] == sigma)]
        if thresfirst:
            stdlist['std1sttruth'] = stdlist['std']
        yerrall = np.vstack([stdlist['stdtruth']-np.sqrt(np.power(stdlist['stdtruth'],2)*stdlist['N']/chi2.ppf(1-alpha, stdlist['N'])), 
                             np.sqrt(np.power(stdlist['stdtruth'],2)*stdlist['N']/chi2.ppf(alpha, stdlist['N']))-stdlist['stdtruth']])
        yerr1st = np.vstack([stdlist['std1sttruth']-np.sqrt(np.power(stdlist['std1sttruth'],2)*stdlist['N']/chi2.ppf(1-alpha, stdlist['N'])), np.sqrt(np.power(stdlist['std1sttruth'],2)*stdlist['N']/chi2.ppf(alpha, stdlist['N']))-stdlist['std1sttruth']])
        ax.errorbar(stdlist['mu'], stdlist['stdtruth'], yerr=yerrall, marker=marker2[i][j], color=colors2[i][j])
        ax.errorbar(stdlist['mu'], stdlist['std1sttruth'], yerr=yerr1st, marker=marker2[i][j], color=colors2[i][j], linestyle='dashed')
ax.set_xlabel(r'$\mu$')
ax.set_ylabel(r'$\mathrm{Time\ resolution}/\si{ns}$')
ax.grid()
# ax.legend(loc='upper right')
fig.savefig('Note/figures/vs-deltadiv.pgf')
fig.savefig('Note/figures/vs-deltadiv.pdf')
fig.savefig('Note/figures/vs-deltadiv.png')
plt.close(fig)

fig = plt.figure(figsize=(5, 4))
gs = gridspec.GridSpec(1, 1, figure=fig, left=0.15, right=0.92, top=0.92, bottom=0.15, wspace=0.3, hspace=0.2)
ax = fig.add_subplot(gs[0, 0])
sigma = 0
tau = max(Tau)
stdlist = mts['firstthres'][(mts['firstthres']['tau'] == Tau[0]) & (mts['firstthres']['sigma'] == Sigma[0])]
yerr = np.vstack([std1sttruth / std1sttruth*(1-1/np.sqrt(stats.f.ppf(1-alpha, stdlist['N']-1, stdlist['N']-1))), (1/np.sqrt(stats.f.ppf(alpha, stdlist['N']-1, stdlist['N']-1))-1)*std1sttruth / std1sttruth])
ax.errorbar(stdlist['mu'], std1sttruth / std1sttruth, yerr=yerr, label=fr'$(20,0)$', marker='o', color='g')
for i, sigma in enumerate(Sigma):
    for j, tau in enumerate(Tau):
        stdlist = mts['firstthres'][(mts['firstthres']['tau'] == tau) & (mts['firstthres']['sigma'] == sigma)]
        if thresfirst:
            stdlist['std1sttruth'] = stdlist['std']
        sigma = int(sigma)
        tau = int(tau)
        yerr = np.vstack([stdlist['std1sttruth'] / stdlist['stdtruth']*(1-1/np.sqrt(stats.f.ppf(1-alpha, stdlist['N']-1, stdlist['N']-1))), (1/np.sqrt(stats.f.ppf(alpha, stdlist['N']-1, stdlist['N']-1))-1)*stdlist['std1sttruth']/stdlist['stdtruth']])
        ax.errorbar(stdlist['mu'], stdlist['std1sttruth'] / stdlist['stdtruth'], yerr=yerr, label=fr'$({tau},{sigma})$', marker=marker2[i][j], color=colors2[i][j])
ax.set_xlabel(r'$\mu$')
# ax.set_ylabel(r'$\mathrm{\sigma_\mathrm{1st}/\sigma_\mathrm{ALL}\ ratio}$')
ax.set_ylabel(r'$\mathrm{ratio}$')
ax.set_ylim(0.9, 3.0)
ax.grid()
# ax.legend(title=r'$(\tau_l, \sigma_l)/\si{ns}$', bbox_to_anchor=(1., 0.9))
ax.legend(title=r'$(\tau_l, \sigma_l)/\si{ns}$')
fig.savefig('Note/figures/vs-deltadiv-r.pgf')
fig.savefig('Note/figures/vs-deltadiv-r.pdf')
fig.savefig('Note/figures/vs-deltadiv-r.png')
plt.close(fig)

# del mts['firstthres']
# keylist = list(mts.keys())
x = np.arange(0, len(keylist) - 1)
mu = 4.0
tau = 20
sigma = 5
wdist = np.vstack([mts[key][(mts[key]['mu'] == mu) & (mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]['wdist'] for key in keylist if key != 'firstthres'])
wdist_dy = np.vstack([wdist[:, 1] - wdist[:, 0], wdist[:, 2] - wdist[:, 1]])
bar_colors = ['b' if key in ['lucyddm', 'takara', 'xiaopeip', 'fsmp'] else 'c' for key in keylist if key != 'firstthres']
labels = ['$'+label[key]+'$' for key in keylist if key != 'firstthres']
keys = [key for key in keylist if key != 'firstthres']
fig = plt.figure(figsize=(10, 5))
fig.tight_layout()
gs = gridspec.GridSpec(1, 1, figure=fig, left=0.1, right=0.9, top=0.9, bottom=0.3, wspace=0., hspace=0.)
ax = fig.add_subplot(gs[0, 0])
ax.bar(x, wdist[:, 1], color=bar_colors)
# ax.set_ylim(0, math.ceil(wdist[~np.isnan(wdist)].max() + 0.5))
ax.set_xlim(-0.5, 7.5)
ax.set_ylim(0, 4)
ax.set_ylabel(r'$\mathrm{Wasserstein\ Distance}/\si{ns}$')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.errorbar(x, wdist[:, 1], yerr=wdist_dy, fmt='o', ecolor='r', c='r', elinewidth=1, capsize=3)
ax.axvline(x=1.5, color='k', linestyle='dashed')
ax.axvline(x=3.5, color='k', linestyle='dashed')
ax.axvline(x=4.5, color='k', linestyle='dashed')
ax2 = ax.twiny()
ax2.set_xticks([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 6.0, 7.5])
ax2.set_xticklabels(['', r'$\mathrm{Heuristic\ methods}$', '', r'$\mathrm{Deconvolution}$', '', '', r'$\mathrm{Regression\ analysis}', ''])
# ax.set_title(fr'$\tau_l={tau}\si{{ns}},\,\sigma_l={sigma}\si{{ns}}$')
# fig.savefig('Note/figures/summarycharge1d.pgf')
# fig.savefig('Note/figures/summarycharge1d.pdf')
fig.savefig('Note/figures/summarycharge1d.png')
fig.clf()
plt.close(fig)

fig = plt.figure(figsize=(8, 6))
fig.tight_layout()
gs = gridspec.GridSpec(1, 1, figure=fig, left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0., hspace=0.)
ax = fig.add_subplot(gs[0, 0])
consumption = np.vstack([mts[key][(mts[key]['mu'] == mu) & (mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]['consumption'] for key in keylist if key != 'firstthres'])
consumption_dy = np.vstack([consumption[:, 1] - consumption[:, 0], consumption[:, 2] - consumption[:, 1]])
for i, (cc, ll, kk) in enumerate(zip(bar_colors, labels, keys)):
    try:
        if kk == 'takara':
            eb = ax.errorbar(consumption[i, 1], wdist[i, 1], xerr=consumption_dy[:, i][:, None], yerr=wdist_dy[:, i][:, None], fmt='o', ecolor=color[kk], c=color[kk], elinewidth=1, capsize=0, label='$\mathrm{CNN(GPU)}$')
            eb[-1][0].set_linestyle('--')
            eb[-1][1].set_linestyle('--')
            with h5py.File('result/takara/solu/' + str(mu) + '-' + str(tau) + '-' + str(sigma) + '.h5', 'r', libver='latest', swmr=True) as soluf:
                time = soluf['starttime_cpu'][:]
                consumption_i = np.insert(np.percentile(time['consumption'], [alpha * 100, 100 - alpha * 100]), 1, time['consumption'].mean())
            consumption_dy_i = np.array([consumption_i[1] - consumption_i[0], consumption_i[2] - consumption_i[1]])
            ax.errorbar(consumption_i[1], wdist[i, 1], xerr=consumption_dy_i[:, None], yerr=wdist_dy[:, i][:, None], fmt='o', elinewidth=1, capsize=3, c=color[kk], label='$\mathrm{CNN(CPU)}$')
        else:
            if kk == 'fsmp':
                eb = ax.errorbar(consumption[i, 1], wdist[i, 1], xerr=consumption_dy[:, i][:, None], yerr=wdist_dy[:, i][:, None], fmt='o', ecolor=color[kk], c=color[kk], elinewidth=1, capsize=3, label='$\mathrm{FSMP(GPU)}$')
                eb[-1][0].set_linestyle('--')
                eb[-1][1].set_linestyle('--')
            else:
                ax.errorbar(consumption[i, 1], wdist[i, 1], xerr=consumption_dy[:, i][:, None], yerr=wdist_dy[:, i][:, None], fmt='o', ecolor=color[kk], c=color[kk], elinewidth=1, capsize=3, label=ll)
    except:
        pass
    ax.text(np.exp(np.log(consumption[i, 1]) + 0.05), wdist[i, 1] + 0.05, s=ll)
min_consumption = np.min(consumption[~np.isnan(consumption)]) / 2
x = np.logspace(np.log10(min_consumption), 2, 301)
ax.plot(x, 2 - 0.5 * x, color='k', alpha=0.5, linestyle='dashed')
ax.fill_between(x, y1=2 - 0.5 * x, y2=10, color='k', alpha=0.2)
ax.set_xlim(min_consumption, 20)
ax.set_ylim(0, 4)
ax.set_xscale('log')
ax.set_xlabel(r'$\mathrm{Time\ consumption\ per\ waveform}/\si{s}$')
ax.set_ylabel(r'$\mathrm{Wasserstein\ Distance}/\si{ns}$')
ax.grid()
# ax.legend()
fig.savefig('Note/figures/summarycharge.pgf')
fig.savefig('Note/figures/summarycharge.pdf')
fig.savefig('Note/figures/summarycharge.png')
fig.clf()
plt.close(fig)

fig_new = plt.figure(figsize=(5, 3))
key = "fsmp"
tau = 20
sigma = 5
ax = fig_new.add_subplot(1, 1, 1)
stdlist = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
yerrcha = np.vstack([-t.ppf(alpha, stdlist['N'])*stdlist['std']/np.sqrt(stdlist['N']), t.ppf(1-alpha, stdlist['N'])*stdlist['std']/np.sqrt(stdlist['N'])])
ax.errorbar(stdlist['mu'] + jitter[key], stdlist['bias'], yerr=yerrcha, label="FSMP", marker=marker[key])
ax.set_xlabel(r"$\mu$")
ax.set_ylabel("bias/ns")
ax.set_title(r"Bias of $t_0$")

fig_new.savefig("bias_t0.pdf", transparent=True, bbox_inches = "tight")
fig_new.savefig("bias_t0.pgf", transparent=True, bbox_inches = "tight")
plt.close(fig_new)

fig_new = plt.figure(figsize=(5, 3))
ax = fig_new.add_subplot(1, 1, 1)
yerr = np.vstack([stdlist['std'] / stdlist['stdtruth']*(1-1/np.sqrt(stats.f.ppf(1-alpha, stdlist['N']-1, stdlist['N']-1))), 
                    (1/np.sqrt(stats.f.ppf(alpha, stdlist['N']-1, stdlist['N']-1))-1)*stdlist['std']/stdlist['stdtruth']])
ax.errorbar(stdlist['mu'] + jitter[key], stdlist['std'] / stdlist['stdtruth'], yerr=yerr, label="FSMP", marker=marker[key])
ax.set_xlabel(r"$\mu$")
ax.set_ylabel("ratio")
ax.set_title(r"Resolution of $t_0$")

fig_new.savefig("ratio_t0.pdf", transparent=True, bbox_inches = "tight")
fig_new.savefig("ratio_t0.pgf", transparent=True, bbox_inches = "tight")
plt.close(fig_new)

for key in mts.keys():
    print(key.rjust(10) + ' stdchargesuccess mean = {:.04%}'.format((mts[key]['stdsuccess'] / mts[key]['N']).mean()))
    print(key.rjust(10) + '  stdchargesuccess min = {:.04%}'.format((mts[key]['stdsuccess'] / mts[key]['N']).min()))
for m in ['mcmc', 'fsmp']:
    print((m + 'one').rjust(10) + '   stdwavesuccess mean = {:.04%}'.format((mts[m]['stdonesuccess'] / mts[m]['N']).mean()))
    print((m + 'one').rjust(10) + '    stdwavesuccess min = {:.04%}'.format((mts[m]['stdonesuccess'] / mts[m]['N']).min()))

stype = np.dtype([('mu', np.float64), ('tau', np.float64), ('sigma', np.float64), ('n', np.uint), ('resnpe', np.float64), ('meanmutru', np.float64), ('stdmutru', np.float64), ('stdmuint', np.float64), ('stdmupe', np.float64), ('stdmumax', np.float64), ('biasmuint', np.float64, 3), ('stdmu', np.float64), ('biasmupe', np.float64, 3), ('biasmumax', np.float64, 3), ('biasmu', np.float64, 3), ('N', np.uint)])
mtsi = np.zeros(len(numbers), dtype=stype)
mtsi['mu'] = np.array([i[0] for i in numbers])
mtsi['tau'] = np.array([i[1] for i in numbers])
mtsi['sigma'] = np.array([i[2] for i in numbers])
mtsi['n'] = np.arange(len(numbers))
mtsi['resnpe'] = np.nan
mtsi['meanmutru'] = np.nan
mtsi['stdmutru'] = np.nan
mtsi['stdmuint'] = np.nan
mtsi['stdmupe'] = np.nan
mtsi['stdmumax'] = np.nan
mtsi['stdmu'] = np.nan
mtsi['biasmuint'] = np.nan
mtsi['biasmupe'] = np.nan
mtsi['biasmumax'] = np.nan
mtsi['biasmu'] = np.nan
mtsi = np.sort(mtsi, kind='stable', order=['mu', 'tau', 'sigma'])

mu_list = np.unique(mts[default_method]['mu'])
mu_std_tru_list = np.sqrt(mu_list)

mts = {'lucyddm':mtsi.copy(), 'takara':mtsi.copy(), 'xiaopeip':mtsi.copy(), 'fsmp':mtsi.copy()}
# mts = {'lucyddm':mtsi.copy(), 'fsmp':mtsi.copy()}

use_log = False
for key in tqdm(mts.keys()):
    for i in range(len(mts[key])):
        f = filelist[mts[key][i]['n']]
        mu = mts[key][i]['mu']
        tau = mts[key][i]['tau']
        sigma = mts[key][i]['sigma']
        waveform = 'waveform'
        try:
            with h5py.File(os.path.join('result', key, 'solu', f), 'r', libver='latest', swmr=True) as soluf, h5py.File(os.path.join('result', key, 'dist', f), 'r', libver='latest', swmr=True) as distf, h5py.File(os.path.join(waveform, f), 'r', libver='latest', swmr=True) as wavef:
                time = soluf['starttime'][:]
                starttime_attrs = dict(soluf['starttime'].attrs)
                start = wavef['SimTruth/T'][:]
                pelist = wavef['SimTriggerInfo/PEList'][:]
                waves = wavef['Readout/Waveform'][:]
                npe_removed = wavef['Readout/Waveform'].attrs['npe_removed']
                wavesum_removed = wavef['Readout/Waveform'].attrs['wavesum_removed']
                mu_true = wavef["Readout/Waveform"].attrs["mu"]
                gmu = wavef['SimTriggerInfo/PEList'].attrs['gmu']
                gsigma = wavef['SimTriggerInfo/PEList'].attrs['gsigma']
                # r = wavef['SimTruth/T'].attrs['r']
                r = np.inf
            vali = np.abs(time['tscharge'] - start['T0'] - np.mean(time['tscharge'] - start['T0'])) <= r * np.std(time['tscharge'] - start['T0'], ddof=-1)
            # vali = np.full(len(time), True)
            Chnum = len(np.unique(pelist['PMTId']))
            e_ans, i_ans = np.unique(pelist['TriggerNo'] * Chnum + pelist['PMTId'], return_index=True)
            i_ans = np.append(i_ans, len(pelist))
            pe_sum = np.array([pelist[i_ans[i]:i_ans[i+1]]['Charge'].sum() for i in range(len(e_ans))]) / gmu
            wave_sum = waves['Waveform'].sum(axis=1) / gmu

            npe = np.diff(i_ans)
            N = len(start) + len(npe_removed)
            N_add = N / (1 - poisson.cdf(0, mu)) - N
            mu_add_fsmp = expon.rvs(size=round(N_add) + len(wavesum_removed), scale=mu_true / (1 + mu_true))
            mu_add = np.hstack([np.zeros(round(N_add)), wavesum_removed / gmu])
            # mu_add = np.hstack([np.zeros(round(N_add)), npe_removed])
            mts[key][i]['N'] = N + round(N_add)
            s_npe = np.sqrt(mu)
            wave_sum_recovered = np.append(wave_sum[vali], mu_add)
            pe_sum_recovered = np.append(pe_sum[vali], mu_add)
            mucharge_recovered = np.append(time['mucharge'][vali], mu_add)
            muwave_recovered = np.append(time['muwave'][vali], mu_add_fsmp)

            # Use sample STD to estimate STD
            mts[key][i]['stdmutru'] = np.sqrt(mu)
            mts[key][i]['meanmutru'] = mu
            mts[key][i]['stdmuint'] = np.std(wave_sum_recovered, ddof=-1)
            # mts[key][i]['biasmuint'] = np.mean(wave_sum_recovered) - mu
            mts[key][i]['biasmuint'] = np.insert(np.percentile(wave_sum_recovered, [alpha * 100, 100 - alpha * 100]), 1, wave_sum_recovered.mean()) - mu
            mts[key][i]['stdmupe'] = np.std(pe_sum_recovered, ddof=-1)
            # mts[key][i]['biasmupe'] = np.mean(pe_sum_recovered) - mu
            mts[key][i]['biasmupe'] = np.insert(np.percentile(pe_sum_recovered, [alpha * 100, 100 - alpha * 100]), 1, pe_sum_recovered.mean()) - mu
            mts[key][i]['stdmumax'] = np.std(mucharge_recovered, ddof=-1)
            # mts[key][i]['biasmumax'] = np.mean(mucharge_recovered) - mu
            mts[key][i]['biasmumax'] = np.insert(np.percentile(mucharge_recovered, [alpha * 100, 100 - alpha * 100]), 1, mucharge_recovered.mean()) - mu
            mts[key][i]['stdmu'] = np.std(muwave_recovered, ddof=-1)
            # mts[key][i]['biasmu'] = np.mean(muwave_recovered) - mu
            mts[key][i]['biasmu'] = np.insert(np.percentile(muwave_recovered, [alpha * 100, 100 - alpha * 100]), 1, muwave_recovered.mean()) - mu

            mts[key][i]['resnpe'] = np.average(mts[key][i]['mu'] / (mts[key][i]['mu'] + 1) * (pe_sum_recovered + 1))
        except:
            pass

keylist = mts.keys()
fig_mu_resolution_ratio = plt.figure(figsize=(len(Tau) * 5, len(Sigma) * 3))
gs_mu_resolution_ratio = gridspec.GridSpec(1, 2, figure=fig_mu_resolution_ratio, left=0.1, right=0.8, top=0.92, bottom=0.15, wspace=0.3, hspace=0.35)
fig_mu_resolution = plt.figure(figsize=(len(Tau) * 5, len(Sigma) * 3))
gs_mu_resolution = gridspec.GridSpec(1, 2, figure=fig_mu_resolution, left=0.1, right=0.8, top=0.92, bottom=0.15, wspace=0.3, hspace=0.35)
fig_mu_bias_ratio = plt.figure(figsize=(len(Tau) * 5, len(Sigma) * 3))
gs_mu_bias_ratio = gridspec.GridSpec(1, 2, figure=fig_mu_bias_ratio, left=0.1, right=0.8, top=0.92, bottom=0.15, wspace=0.3, hspace=0.35)
fig_mu_bias = plt.figure(figsize=(len(Tau) * 5, len(Sigma) * 3))
gs_mu_bias = gridspec.GridSpec(1, 2, figure=fig_mu_bias, left=0.1, right=0.8, top=0.92, bottom=0.15, wspace=0.3, hspace=0.35)
for i, sigma in enumerate(Sigma):
    for j, tau in enumerate(Tau):
        #
        # related intensity resolution comparison
        #
        stdlist = mts[default_method][(mts[default_method]['tau'] == tau) & (mts[default_method]['sigma'] == sigma)]
        ax = fig_mu_resolution_ratio.add_subplot(gs_mu_resolution_ratio[i, j])
        yerr = np.vstack([stdlist['stdmuint']-np.sqrt(np.power(stdlist['stdmuint'],2)*stdlist['N']/chi2.ppf(1-alpha, stdlist['N'])), 
                          np.sqrt(np.power(stdlist['stdmuint'],2)*stdlist['N']/chi2.ppf(alpha, stdlist['N']))-stdlist['stdmuint']]) / (stdlist['biasmuint'][:, 1] + stdlist['meanmutru'])
        #ax.errorbar(stdlist['mu'] + jitter['tru'], 
        #            stdlist['stdmuint'] / (stdlist['biasmuint'][:, 1] + stdlist['meanmutru']) / (1 / np.sqrt(stdlist['mu'])),
        #            yerr=yerr / (1 / np.sqrt(stdlist['mu'])), 
        #            label='$\mathrm{int}$', 
        #            c=color['1st'], 
        #            marker=marker['1st'])
        for key in keylist:
            stdlist = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
            yerr = np.vstack([stdlist['stdmu']-np.sqrt(np.power(stdlist['stdmu'],2)*stdlist['N']/chi2.ppf(1-alpha, stdlist['N'])), 
                              np.sqrt(np.power(stdlist['stdmu'],2)*stdlist['N']/chi2.ppf(alpha, stdlist['N']))-stdlist['stdmu']]) / (stdlist['biasmu'][:, 1] + stdlist['meanmutru'])
            ax.errorbar(stdlist['mu'] + jitter[key], 
                        stdlist['stdmu'] / (stdlist['biasmu'][:, 1] + stdlist['meanmutru']) / (1 / np.sqrt(stdlist['mu'])), 
                        yerr=yerr / (1 / np.sqrt(stdlist['mu'])), 
                        label='$' + label[key] + '$', 
                        c=color[key], 
                        marker=marker[key])
        ax.axhline(y=1, color='k', alpha=0.5)
        ax.set_xlabel(r'$\mu$')
        ax.set_ylabel(r'$\mathrm{ratio}$')
        ax.set_title(fr'$\tau_l={tau}\si{{ns}},\,\sigma_l={sigma}\si{{ns}}$')
        # ax.set_ylim(0.96, 1.54)
        ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
        ax.grid()
        if i == len(Sigma) - 1 and j == len(Tau) - 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1., 0.9))

        #
        # intensity resolution comparison
        #
        stdlist = mts[default_method][(mts[default_method]['tau'] == tau) & (mts[default_method]['sigma'] == sigma)]
        ax = fig_mu_resolution.add_subplot(gs_mu_resolution[i, j])
        yerr = np.vstack([stdlist['stdmuint']-np.sqrt(np.power(stdlist['stdmuint'],2)*stdlist['N']/chi2.ppf(1-alpha, stdlist['N'])), 
                          np.sqrt(np.power(stdlist['stdmuint'],2)*stdlist['N']/chi2.ppf(alpha, stdlist['N']))-stdlist['stdmuint']]) / (stdlist['biasmuint'][:, 1] + stdlist['meanmutru'])
        #ax.errorbar(stdlist['mu'] + jitter['tru'], stdlist['stdmuint'] / (stdlist['biasmuint'][:, 1] + stdlist['meanmutru']), yerr=yerr, label='$\mathrm{int}$', c=color['1st'], marker=marker['1st'])
        for key in keylist:
            stdlist = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
            yerr = np.vstack([stdlist['stdmu']-np.sqrt(np.power(stdlist['stdmu'],2)*stdlist['N']/chi2.ppf(1-alpha, stdlist['N'])), np.sqrt(np.power(stdlist['stdmu'],2)*stdlist['N']/chi2.ppf(alpha, stdlist['N']))-stdlist['stdmu']]) / (stdlist['biasmu'][:, 1] + stdlist['meanmutru'])
            ax.errorbar(stdlist['mu'] + jitter[key], stdlist['stdmu'] / (stdlist['biasmu'][:, 1] + stdlist['meanmutru']), yerr=yerr, label='$' + label[key] + '$', c=color[key], marker=marker[key])
        ax.plot(mu_list, np.sqrt(mu_std_tru_list**2 + (gsigma / gmu)**2) / mu_list, color='k', alpha=0.5)
        ax.set_xlabel(r'$\mu$')
        # ax.set_ylabel(r'$\mathrm{ratio}$')
        ax.set_ylabel(r'$\sigma_{\hat{\mu}}/\hat{\mu}$')
        ax.set_title(fr'$\tau_l={tau}\si{{ns}},\,\sigma_l={sigma}\si{{ns}}$')
        # ax.set_ylim(0.96, 1.54)
        ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
        ax.grid()
        if i == len(Sigma) - 1 and j == len(Tau) - 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1., 0.9))

        #
        # relative charge bias
        #
        stdlist = mts[default_method][(mts[default_method]['tau'] == tau) & (mts[default_method]['sigma'] == sigma)]
        ax = fig_mu_bias_ratio.add_subplot(gs_mu_bias_ratio[i, j])
        yerr = np.vstack([-t.ppf(alpha, stdlist['N'])*stdlist['stdmuint']/np.sqrt(stdlist['N']), 
                          t.ppf(1-alpha, stdlist['N'])*stdlist['stdmuint']/np.sqrt(stdlist['N'])]) / stdlist['meanmutru']
        #ax.errorbar(stdlist['mu'] + jitter['tru'], stdlist['biasmuint'][:, 1] / stdlist['meanmutru'], yerr=yerr, label='$\mathrm{int}$', c=color['1st'], marker=marker['1st'])

        # yerr = np.vstack([stdlist['biasmuint'][:, 1] - stdlist['biasmuint'][:, 0], stdlist['biasmuint'][:, 2] - stdlist['biasmuint'][:, 1]])
        # ax.errorbar(stdlist['mu'] + jitter['tru'], stdlist['biasmuint'][:, 1] / stdlist['meanmutru'], yerr=yerr / stdlist['meanmutru'], label='$\mathrm{int}$', c=color['1st'], marker=marker['1st'])

        for key in keylist:
            stdlist = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
            yerr = np.vstack([-t.ppf(alpha, stdlist['N'])*stdlist['stdmu']/np.sqrt(stdlist['N']), t.ppf(1-alpha, stdlist['N'])*stdlist['stdmu']/np.sqrt(stdlist['N'])]) / stdlist['meanmutru']
            ax.errorbar(stdlist['mu'] + jitter[key], stdlist['biasmu'][:, 1] / stdlist['meanmutru'], yerr=yerr, label='$' + label[key] + '$', c=color[key], marker=marker[key])
            # yerr = np.vstack([stdlist['biasmu'][:, 1] - stdlist['biasmu'][:, 0], stdlist['biasmu'][:, 2] - stdlist['biasmu'][:, 1]])
            # ax.errorbar(stdlist['mu'] + jitter['tru'], stdlist['biasmu'][:, 1] / stdlist['meanmutru'], yerr=yerr / stdlist['meanmutru'], label='$' + label[key] + '$', c=color[key], marker=marker[key])
        ax.set_xlabel(r'$\mu$')
        if use_log:
            ax.set_ylabel(r'$\frac{\Delta \log\mu}{\log\mu}$')
        else:
            ax.set_ylabel(r'$\mathrm{bias}$')
        ax.set_title(fr'$\tau_l={tau}\si{{ns}},\,\sigma_l={sigma}\si{{ns}}$')
        # ax.set_ylim(-0.03, 0.025)
        ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
        ax.grid()
        if i == len(Sigma) - 1 and j == len(Tau) - 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1., 0.9))

        #
        # charge bias
        #
        stdlist = mts[default_method][(mts[default_method]['tau'] == tau) & (mts[default_method]['sigma'] == sigma)]
        ax = fig_mu_bias.add_subplot(gs_mu_bias[i, j])
        yerr = np.vstack([-t.ppf(alpha, stdlist['N'])*stdlist['stdmuint']/np.sqrt(stdlist['N']), t.ppf(1-alpha, stdlist['N'])*stdlist['stdmuint']/np.sqrt(stdlist['N'])])
        #ax.errorbar(stdlist['mu'] + jitter['tru'], stdlist['biasmuint'][:, 1], yerr=yerr, label='$\mathrm{int}$', c=color['1st'], marker=marker['1st'])

        # yerr = np.vstack([stdlist['biasmuint'][:, 1] - stdlist['biasmuint'][:, 0], stdlist['biasmuint'][:, 2] - stdlist['biasmuint'][:, 1]])
        # ax.errorbar(stdlist['mu'] + jitter['tru'], stdlist['biasmuint'][:, 1], yerr=yerr, label='$\mathrm{int}$', c=color['1st'], marker=marker['1st'])

        for key in keylist:
            stdlist = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
            yerr = np.vstack([-t.ppf(alpha, stdlist['N'])*stdlist['stdmu']/np.sqrt(stdlist['N']), t.ppf(1-alpha, stdlist['N'])*stdlist['stdmu']/np.sqrt(stdlist['N'])])
            ax.errorbar(stdlist['mu'] + jitter[key], stdlist['biasmu'][:, 1], yerr=yerr, label='$' + label[key] + '$', c=color[key], marker=marker[key])
            # yerr = np.vstack([stdlist['biasmu'][:, 1] - stdlist['biasmu'][:, 0], stdlist['biasmu'][:, 2] - stdlist['biasmu'][:, 1]])
            # ax.errorbar(stdlist['mu'] + jitter['tru'], stdlist['biasmu'][:, 1], yerr=yerr, label='$' + label[key] + '$', c=color[key], marker=marker[key])
        ax.set_xlabel(r'$\mu$')
        if use_log:
            ax.set_ylabel(r'$\Delta \log\mu$')
        else:
            ax.set_ylabel(r'$\mathrm{bias}$')
        ax.set_title(fr'$\tau_l={tau}\si{{ns}},\,\sigma_l={sigma}\si{{ns}}$')
        # ax.set_ylim(-0.056, 0.048)
        ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
        ax.grid()
        if i == len(Sigma) - 1 and j == len(Tau) - 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1., 0.9))
fig_mu_resolution_ratio.savefig('Note/figures/vs-sigmamu-r.pgf') # Ratio of mu's resolution
fig_mu_resolution_ratio.savefig('Note/figures/vs-sigmamu-r.pdf')
fig_mu_resolution_ratio.savefig('Note/figures/vs-sigmamu-r.png')
plt.close(fig_mu_resolution_ratio)
fig_mu_resolution.savefig('Note/figures/vs-sigmamu.pgf')
fig_mu_resolution.savefig('Note/figures/vs-sigmamu.pdf')
fig_mu_resolution.savefig('Note/figures/vs-sigmamu.png')
plt.close(fig_mu_resolution)
fig_mu_bias_ratio.savefig('Note/figures/vs-biasmu-r.pgf') # Relative mu biases
fig_mu_bias_ratio.savefig('Note/figures/vs-biasmu-r.pdf')
fig_mu_bias_ratio.savefig('Note/figures/vs-biasmu-r.png')
plt.close(fig_mu_bias_ratio)
fig_mu_bias.savefig('Note/figures/vs-biasmu.pgf')
fig_mu_bias.savefig('Note/figures/vs-biasmu.pdf')
fig_mu_bias.savefig('Note/figures/vs-biasmu.png')
plt.close(fig_mu_bias)

fig_new = plt.figure(figsize=(5, 3))
key = "fsmp"
tau = 20
sigma = 5
ax = fig_new.add_subplot(1, 1, 1)
stdlist = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
yerr = np.vstack([-t.ppf(alpha, stdlist['N'])*stdlist['stdmu']/np.sqrt(stdlist['N']), t.ppf(1-alpha, stdlist['N'])*stdlist['stdmu']/np.sqrt(stdlist['N'])]) / stdlist['meanmutru']
ax.errorbar(stdlist['mu'] + jitter[key], stdlist['biasmu'][:, 1] / stdlist['meanmutru'], yerr=yerr, label="FSMP", marker=marker[key])
ax.set_xlabel(r"$\mu$")
ax.set_ylabel("bias")
ax.set_title(r"Relative bias of $\mu$")

fig_new.savefig("bias_mu.pdf", transparent=True, bbox_inches = "tight")
fig_new.savefig("bias_mu.pgf", transparent=True, bbox_inches = "tight")
plt.close(fig_new)

fig_new = plt.figure(figsize=(5, 3))
ax = fig_new.add_subplot(1, 1, 1)
yerr = np.vstack([stdlist['stdmu']-np.sqrt(np.power(stdlist['stdmu'],2)*stdlist['N']/chi2.ppf(1-alpha, stdlist['N'])), 
                    np.sqrt(np.power(stdlist['stdmu'],2)*stdlist['N']/chi2.ppf(alpha, stdlist['N']))-stdlist['stdmu']]) / (stdlist['biasmu'][:, 1] + stdlist['meanmutru'])
ax.errorbar(stdlist['mu'] + jitter[key], 
            stdlist['stdmu'] / (stdlist['biasmu'][:, 1] + stdlist['meanmutru']) / stdlist['resnpe'], 
            yerr=yerr / (1 / np.sqrt(stdlist['mu'])), 
            label="FSMP", 
            marker=marker[key])
ax.set_xlabel(r"$\mu$")
ax.set_ylabel("ratio")
ax.set_title(r"Resolution of $\mu$")

fig_new.savefig("ratio_mu.pdf", transparent=True, bbox_inches = "tight")
fig_new.savefig("ratio_mu.pgf", transparent=True, bbox_inches = "tight")
plt.close(fig_new)
