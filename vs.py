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
from scipy.stats import norm, poisson, uniform, chi2, t
from scipy.integrate import quad
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec

import wf_func as wff

import matplotlib
matplotlib.use('pgf')

plt.rcParams['font.size'] = 12

psr = argparse.ArgumentParser()
psr.add_argument('--conf', type=str, help='configuration of tau & sigma')
args = psr.parse_args()

with open(args.conf) as f:
    f_csv = csv.reader(f, delimiter=' ')
    Tau = next(f_csv)
    Tau = [float(i) for i in Tau]
    Sigma = next(f_csv)
    Sigma = [float(i) for i in Sigma]

filelist = os.listdir('result/fbmp/solu')
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

mts = {'firstthres':mtsi.copy(), 'threshold':mtsi.copy(), 'findpeak':mtsi.copy(), 'fftrans':mtsi.copy(), 'lucyddm':mtsi.copy(), 'takara':mtsi.copy(), 'xiaopeip':mtsi.copy(), 'mcmc':mtsi.copy(), 'fbmp':mtsi.copy()}
deltalabel = {'1st':'\mathrm{1st}', 'tru':'\mathrm{ALL}', 'findpeak':'\mathrm{FindPeak}', 'threshold':'\mathrm{Shift}', 'fftrans':'\mathrm{FFT}', 'lucyddm':'\mathrm{LucyDDM}', 'xiaopeip':'\mathrm{Fitting}', 'takara':'\mathrm{CNN}', 'fbmp':'\mathrm{FBMP}', 'fbmpone':'\mathrm{FBMPmax}', 'mcmc':'\mathrm{MCMCt0}', 'mcmcone':'\mathrm{MCMCcha}', 'firstthres':'\mathrm{1stthres}'}
label = {'1st':'\mathrm{1st}', 'tru':'\mathrm{ALL}', 'findpeak':'\mathrm{Peak\ finding}', 'threshold':'\mathrm{Waveform\ shifting}', 'fftrans':'\mathrm{Fourier\ deconvolution}', 'lucyddm':'\mathrm{LucyDDM}', 'xiaopeip':'\mathrm{Fitting}', 'takara':'\mathrm{CNN}', 'fbmp':'\mathrm{FBMP}', 'fbmpone':'\mathrm{FBMPmax}', 'mcmc':'\mathrm{MCMC}', 'firstthres':'\mathrm{1stthres}'}
marker = {'1st':'o', 'tru':'h', 'findpeak':',', 'threshold':'1', 'fftrans':'+', 'lucyddm':'p', 'xiaopeip':'*', 'takara':'x', 'fbmp':'s', 'fbmpone':'^', 'mcmc':'X', 'mcmcone':'>', 'firstthres':'>'}
color = {'1st':'g', 'tru':'k', 'findpeak':'C1', 'threshold':'C2', 'fftrans':'m', 'lucyddm':'y', 'xiaopeip':'c', 'takara':'C0', 'fbmp':'r', 'fbmpone':'b', 'mcmc':'C4', 'mcmcone':'C5', 'firstthres':'C6'}
jit = 0.05
jitter = {'mcmcone':-5 * jit, 'mcmc':-4 * jit, 'tru':-3 * jit, 'fbmp':-2 * jit, 'fbmpone':-1 * jit, 'lucyddm':0 * jit, 'xiaopeip':1 * jit, 'takara':2 * jit, '1st':3 * jit, 'fftrans':4 * jit, 'findpeak':5 * jit, 'threshold':6 * jit, 'firstthres':7 * jit}

alpha = 0.05
for key in tqdm(mts.keys()):
    for i in range(len(mts[key])):
        f = filelist[mts[key][i]['n']]
        mu = mts[key][i]['mu']
        tau = mts[key][i]['tau']
        sigma = mts[key][i]['sigma']
        try:
            with h5py.File(os.path.join('result', key, 'solu', f), 'r', libver='latest', swmr=True) as soluf, h5py.File(os.path.join('result', key, 'dist', f), 'r', libver='latest', swmr=True) as distf, h5py.File(os.path.join('waveform', f), 'r', libver='latest', swmr=True) as wavef:
                time = soluf['starttime'][:]
                record = distf['Record'][:]
                start = wavef['SimTruth/T'][:]
                gmu = wavef['SimTriggerInfo/PEList'].attrs['gmu']
                gsigma = wavef['SimTriggerInfo/PEList'].attrs['gsigma']
                r = wavef['SimTruth/T'].attrs['r']
                r = np.inf
            mts[key][i]['N'] = len(start)
            vali = np.abs(time['tscharge'] - start['T0'] - np.mean(time['tscharge'] - start['T0'])) <= r * np.std(time['tscharge'] - start['T0'], ddof=-1)
            # vali = np.full(start['T0'].shape, True)
            mts[key][i]['std1sttruth'] = np.std(start['ts1sttruth'] - start['T0'], ddof=-1)
            mts[key][i]['stdtruth'] = np.std(start['tstruth'] - start['T0'], ddof=-1)
            mts[key][i]['std'], mts[key][i]['stdsuccess'] = wff.stdrmoutlier(time['tscharge'] - start['T0'], r)
            mts[key][i]['bias1sttruth'] = np.mean(start['ts1sttruth'] - start['T0'])
            mts[key][i]['biastruth'] = np.mean(start['tstruth'] - start['T0'])
            mts[key][i]['bias'] = np.mean(time['tscharge'][vali] - start['T0'][vali])
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

lim = {'deltadiv':np.array([[0.3, 0.5]]), 'wdist':np.array([[1.0, 1.5]]), 'rss':np.array([[100, 200]])}

figd = plt.figure(figsize=(len(Tau) * 5, len(Sigma) * 3))
gsd = gridspec.GridSpec(len(Sigma), len(Tau), figure=figd, left=0.1, right=0.8, top=0.92, bottom=0.15, wspace=0.3, hspace=0.35)
figb = plt.figure(figsize=(len(Tau) * 5, len(Sigma) * 3))
gsb = gridspec.GridSpec(len(Sigma), len(Tau), figure=figd, left=0.1, right=0.8, top=0.92, bottom=0.15, wspace=0.3, hspace=0.35)
figdd = plt.figure(figsize=(len(Tau) * 5, len(Sigma) * 3))
gsdd = gridspec.GridSpec(len(Sigma), len(Tau), figure=figd, left=0.1, right=0.8, top=0.92, bottom=0.15, wspace=0.3, hspace=0.35)
figw = plt.figure(figsize=(len(Tau) * 5, len(Sigma) * 3))
gsw = gridspec.GridSpec(len(Sigma), len(Tau), figure=figw, left=0.1, right=0.8, top=0.92, bottom=0.15, wspace=0.3, hspace=0.35)
figr = plt.figure(figsize=(len(Tau) * 5, len(Sigma) * 3))
gsr = gridspec.GridSpec(len(Sigma), len(Tau), figure=figr, left=0.1, right=0.8, top=0.92, bottom=0.15, wspace=0.3, hspace=0.35)
keylist = list(mts.keys())
badkey = ['findpeak', 'threshold', 'fftrans', 'mcmc', 'firstthres']
for i, sigma in enumerate(Sigma):
    for j, tau in enumerate(Tau):
        ax = figd.add_subplot(gsd[i, j])
        stdlist = mts['lucyddm'][(mts['lucyddm']['tau'] == tau) & (mts['lucyddm']['sigma'] == sigma)]
        yerr1st = np.vstack([stdlist['std1sttruth']-np.sqrt(np.power(stdlist['std1sttruth'],2)*stdlist['N']/chi2.ppf(1-alpha/2, stdlist['N'])), np.sqrt(np.power(stdlist['std1sttruth'],2)*stdlist['N']/chi2.ppf(alpha/2, stdlist['N']))-stdlist['std1sttruth']])
        yerrall = np.vstack([stdlist['stdtruth']-np.sqrt(np.power(stdlist['stdtruth'],2)*stdlist['N']/chi2.ppf(1-alpha/2, stdlist['N'])), np.sqrt(np.power(stdlist['stdtruth'],2)*stdlist['N']/chi2.ppf(alpha/2, stdlist['N']))-stdlist['stdtruth']])
        ax.errorbar(stdlist['mu'] + jitter['1st'], stdlist['std1sttruth'], yerr=yerr1st, c=color['1st'], label='$'+deltalabel['1st']+'$', marker=marker['1st'])
        ax.errorbar(stdlist['mu'] + jitter['tru'], stdlist['stdtruth'], yerr=yerrall, c=color['tru'], label='$'+deltalabel['tru']+'$', marker=marker['tru'])
        for m in ['mcmc', 'fbmp']:
            stdlistwav = mts[m][(mts[m]['tau'] == tau) & (mts[m]['sigma'] == sigma)]
            yerrwav = np.vstack([stdlistwav['stdone']-np.sqrt(np.power(stdlistwav['stdone'],2)*stdlistwav['N']/chi2.ppf(1-alpha/2, stdlistwav['N'])), np.sqrt(np.power(stdlistwav['stdone'],2)*stdlistwav['N']/chi2.ppf(alpha/2, stdlistwav['N']))-stdlistwav['stdone']])
            ax.errorbar(stdlistwav['mu'] + jitter[m + 'one'], stdlistwav['stdone'], yerr=yerrwav, c=color[m + 'one'], label='$'+deltalabel[m + 'one']+'$', marker=marker[m + 'one'])
        for key in keylist:
            stdlist = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
            yerrcha = np.vstack([stdlist['std']-np.sqrt(np.power(stdlist['std'],2)*stdlist['N']/chi2.ppf(1-alpha/2, stdlist['N'])), np.sqrt(np.power(stdlist['std'],2)*stdlist['N']/chi2.ppf(alpha/2, stdlist['N']))-stdlist['std']])
            ax.errorbar(stdlist['mu'] + jitter[key], stdlist['std'], yerr=yerrcha, c=color[key], label='$'+deltalabel[key]+'$', marker=marker[key])
        ax.set_xlabel(r'$\mu$')
        ax.set_ylabel(r'$\sigma/\si{ns}$')
        ax.set_title(fr'$\tau_l={tau}\si{{ns}},\,\sigma_l={sigma}\si{{ns}}$')
        # ax.set_yscale('log')
        ax.grid()
        if i == len(Sigma) - 1 and j == len(Tau) - 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1., 0.9), prop={'size': 5})

        ax = figb.add_subplot(gsb[i, j])
        stdlist = mts['lucyddm'][(mts['lucyddm']['tau'] == tau) & (mts['lucyddm']['sigma'] == sigma)]
        # yerr1st = np.vstack([stdlist['bias1sttruth']-t.ppf(1-alpha/2, stdlist['N'])*stdlist['std1sttruth']/np.sqrt(stdlist['N']), t.ppf(1-alpha/2, stdlist['N'])*stdlist['std1sttruth']/np.sqrt(stdlist['N'])-stdlist['bias1sttruth']])
        yerrall = np.vstack([stdlist['biastruth']-t.ppf(1-alpha/2, stdlist['N'])*stdlist['stdtruth']/np.sqrt(stdlist['N']), t.ppf(1-alpha/2, stdlist['N'])*stdlist['stdtruth']/np.sqrt(stdlist['N'])-stdlist['biastruth']])
        # ax.errorbar(stdlist['mu'] + jitter['1st'], stdlist['bias1sttruth'], yerr=yerr1st, c=color['1st'], label='$'+deltalabel['1st']+'$', marker=marker['1st'])
        ax.errorbar(stdlist['mu'] + jitter['tru'], stdlist['biastruth'], yerr=yerrall, c=color['tru'], label='$'+deltalabel['tru']+'$', marker=marker['tru'])
        for key in keylist:
            if key in badkey:
                continue
            stdlist = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
            yerrcha = np.vstack([stdlist['bias']-t.ppf(1-alpha/2, stdlist['N'])*stdlist['std']/np.sqrt(stdlist['N']), t.ppf(1-alpha/2, stdlist['N'])*stdlist['std']/np.sqrt(stdlist['N'])-stdlist['bias']])
            ax.errorbar(stdlist['mu'] + jitter[key], stdlist['bias'], yerr=yerrcha, c=color[key], label='$'+deltalabel[key]+'$', marker=marker[key])
        ax.set_ylim(-0.5, 12)
        ax.set_xlabel(r'$\mu$')
        ax.set_ylabel(r'$\mathrm{bias}/\si{ns}$')
        ax.set_title(fr'$\tau_l={tau}\si{{ns}},\,\sigma_l={sigma}\si{{ns}}$')
        # ax.set_yscale('log')
        ax.grid()
        if i == len(Sigma) - 1 and j == len(Tau) - 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1., 0.9))

        ax = figdd.add_subplot(gsdd[i, j])
        stdlist = mts['lucyddm'][(mts['lucyddm']['tau'] == tau) & (mts['lucyddm']['sigma'] == sigma)]
        for key in keylist:
            if key in badkey:
                continue
            stdlist = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
            # if key == 'fbmp':
            #     yerr = stdlist['stdone'] / stdlist['stdtruth'] / np.sqrt(stdlist['N'])
            #     ax.errorbar(stdlist['mu'], stdlist['stdone'] / stdlist['stdtruth'], yerr=yerr, label='$'+deltalabel['fbmpone']+'$', c=color['fbmpone'], marker=marker['fbmpone'])
            yerr = stdlist['std'] / stdlist['stdtruth'] / np.sqrt(stdlist['N'])
            ax.errorbar(stdlist['mu'] + jitter[key], stdlist['std'] / stdlist['stdtruth'], yerr=yerr, label='$'+deltalabel[key]+'$', c=color[key], marker=marker[key])
        ax.set_xlabel(r'$\mu$')
        ax.set_ylabel(r'$\mathrm{ratio}$')
        ax.set_title(fr'$\tau_l={tau}\si{{ns}},\,\sigma_l={sigma}\si{{ns}}$')
        # ax.set_ylim(lim['deltadiv'][i, j], 1.01)
        ax.grid()
        if i == len(Sigma) - 1 and j == len(Tau) - 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1., 0.9))

        ax = figw.add_subplot(gsw[i, j])
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

        ax = figr.add_subplot(gsr[i, j])
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
figd.savefig('Note/figures/vs-delta.pgf')
figd.savefig('Note/figures/vs-delta.pdf')
figd.savefig('Note/figures/vs-delta.png')
plt.close(figd)
figb.savefig('Note/figures/vs-bias.pgf')
figb.savefig('Note/figures/vs-bias.pdf')
figb.savefig('Note/figures/vs-bias.png')
plt.close(figb)
figdd.savefig('Note/figures/vs-deltamethodsdiv.pgf')
figdd.savefig('Note/figures/vs-deltamethodsdiv.pdf')
figdd.savefig('Note/figures/vs-deltamethodsdiv.png')
plt.close(figdd)
figw.savefig('Note/figures/vs-wdist.pgf')
figw.savefig('Note/figures/vs-wdist.pdf')
figw.savefig('Note/figures/vs-wdist.png')
plt.close(figw)
figr.savefig('Note/figures/vs-rss.pgf')
figr.savefig('Note/figures/vs-rss.pdf')
figr.savefig('Note/figures/vs-rss.png')
plt.close(figr)

thresfirst = False
marker2 = [['s', '^']]
colors2 = [['r', 'b']]
fig = plt.figure(figsize=(8, 4))
gs = gridspec.GridSpec(1, 2, figure=fig, left=0.1, right=0.85, top=0.92, bottom=0.15, wspace=0.25, hspace=0.2)
ax = fig.add_subplot(gs[0, 0])
std1sttruth = np.empty(len(stdlist['mu']))
stdlist = mts['firstthres'][(mts['firstthres']['tau'] == Tau[0]) & (mts['firstthres']['sigma'] == Sigma[0])]
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
yerr = np.vstack([std1sttruth-np.sqrt(np.power(std1sttruth,2)*stdlist['N']/chi2.ppf(1-alpha/2, stdlist['N'])), np.sqrt(np.power(std1sttruth,2)*stdlist['N']/chi2.ppf(alpha/2, stdlist['N']))-std1sttruth])
ax.errorbar(stdlist['mu'], std1sttruth, yerr=yerr, label='$\sigma_'+deltalabel['tru']+fr',\,\tau_l={tau}\si{{ns}},\,\sigma_l={sigma}\si{{ns}}$', marker='o', color='g')
ax.errorbar(stdlist['mu'], std1sttruth, yerr=yerr, label='$\sigma_'+deltalabel['1st']+fr',\,\tau_l={tau}\si{{ns}},\,\sigma_l={sigma}\si{{ns}}$', marker='o', color='g', linestyle='dashed')
for i, sigma in enumerate(Sigma):
    for j, tau in enumerate(Tau):
        stdlist = mts['firstthres'][(mts['firstthres']['tau'] == tau) & (mts['firstthres']['sigma'] == sigma)]
        if thresfirst:
            stdlist['std1sttruth'] = stdlist['std']
        yerrall = np.vstack([stdlist['stdtruth']-np.sqrt(np.power(stdlist['stdtruth'],2)*stdlist['N']/chi2.ppf(1-alpha/2, stdlist['N'])), np.sqrt(np.power(stdlist['stdtruth'],2)*stdlist['N']/chi2.ppf(alpha/2, stdlist['N']))-stdlist['stdtruth']])
        yerr1st = np.vstack([stdlist['std1sttruth']-np.sqrt(np.power(stdlist['std1sttruth'],2)*stdlist['N']/chi2.ppf(1-alpha/2, stdlist['N'])), np.sqrt(np.power(stdlist['std1sttruth'],2)*stdlist['N']/chi2.ppf(alpha/2, stdlist['N']))-stdlist['std1sttruth']])
        ax.errorbar(stdlist['mu'], stdlist['stdtruth'], yerr=yerrall, marker=marker2[i][j], color=colors2[i][j])
        ax.errorbar(stdlist['mu'], stdlist['std1sttruth'], yerr=yerr1st, marker=marker2[i][j], color=colors2[i][j], linestyle='dashed')
ax.set_xlabel(r'$\mu$')
ax.set_ylabel(r'$\mathrm{Time\ resolution}/\si{ns}$')
ax.grid()
# ax.legend(loc='upper right')
ax = fig.add_subplot(gs[0, 1])
sigma = 0
tau = max(Tau)
stdlist = mts['firstthres'][(mts['firstthres']['tau'] == Tau[0]) & (mts['firstthres']['sigma'] == Sigma[0])]
yerr = std1sttruth / std1sttruth / np.sqrt(stdlist['N'])
ax.errorbar(stdlist['mu'], std1sttruth / std1sttruth, yerr=yerr, label=fr'$(20,0)$', marker='o', color='g')
for i, sigma in enumerate(Sigma):
    for j, tau in enumerate(Tau):
        stdlist = mts['firstthres'][(mts['firstthres']['tau'] == tau) & (mts['firstthres']['sigma'] == sigma)]
        if thresfirst:
            stdlist['std1sttruth'] = stdlist['std']
        sigma = int(sigma)
        tau = int(tau)
        yerr = stdlist['std1sttruth'] / stdlist['stdtruth'] / np.sqrt(stdlist['N'])
        ax.errorbar(stdlist['mu'], stdlist['std1sttruth'] / stdlist['stdtruth'], yerr=yerr, label=fr'$({tau},{sigma})$', marker=marker2[i][j], color=colors2[i][j])
ax.set_xlabel(r'$\mu$')
ax.set_ylabel(r'$\mathrm{\sigma_\mathrm{1st}/\sigma_\mathrm{ALL}\ ratio}$')
ax.set_ylim(0.9, 3.0)
ax.grid()
ax.legend(title=r'$(\tau_l, \sigma_l)/\si{ns}$', bbox_to_anchor=(1., 0.9))
fig.savefig('Note/figures/vs-deltadiv.pgf')
fig.savefig('Note/figures/vs-deltadiv.pdf')
fig.savefig('Note/figures/vs-deltadiv.png')
plt.close(fig)

# del mts['firstthres']
# keylist = list(mts.keys())
x = np.arange(0, len(keylist) - 1)
mu = 4.0
tau = 20
sigma = 5
wdist = np.vstack([mts[key][(mts[key]['mu'] == mu) & (mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]['wdist'] for key in keylist if key != 'firstthres'])
wdist_dy = np.vstack([wdist[:, 1] - wdist[:, 0], wdist[:, 2] - wdist[:, 1]])
bar_colors = ['b' if key in ['lucyddm', 'takara', 'xiaopeip', 'fbmp'] else 'c' for key in keylist if key != 'firstthres']
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
        ax.errorbar(consumption[i, 1], wdist[i, 1], xerr=consumption_dy[:, i][:, None], yerr=wdist_dy[:, i][:, None], fmt='o', ecolor=color[kk], c=color[kk], elinewidth=1, capsize=3, label=ll)
    ax.text(np.exp(np.log(consumption[i, 1]) + 0.05), wdist[i, 1] + 0.05, s=ll)
ax.plot(np.logspace(-5, 2, 301), 2 - np.logspace(-5, 2, 301), color='k', alpha=0.5, linestyle='dashed')
ax.fill_between(np.logspace(-5, 2, 301), y1=2 - np.logspace(-5, 2, 301), y2=10, color='k', alpha=0.2)
ax.set_xlim(1e-3, 20)
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

for key in mts.keys():
    print(key.rjust(10) + ' stdchargesuccess mean = {:.04%}'.format((mts[key]['stdsuccess'] / mts[key]['N']).mean()))
    print(key.rjust(10) + '  stdchargesuccess min = {:.04%}'.format((mts[key]['stdsuccess'] / mts[key]['N']).min()))
for m in ['mcmc', 'fbmp']:
    print((m + 'one').rjust(10) + '   stdwavesuccess mean = {:.04%}'.format((mts[m]['stdonesuccess'] / mts[m]['N']).mean()))
    print((m + 'one').rjust(10) + '    stdwavesuccess min = {:.04%}'.format((mts[m]['stdonesuccess'] / mts[m]['N']).min()))

stype = np.dtype([('mu', np.float64), ('tau', np.float64), ('sigma', np.float64), ('n', np.uint), ('meanmutru', np.float64), ('stdmutru', np.float64), ('stdmuint', np.float64), ('stdmupe', np.float64), ('stdmumax', np.float64), ('stdmu', np.float64), ('biasmuint', np.float64), ('biasmupe', np.float64), ('biasmumax', np.float64), ('biasmu', np.float64), ('N', np.uint)])
mtsi = np.zeros(len(numbers), dtype=stype)
mtsi['mu'] = np.array([i[0] for i in numbers])
mtsi['tau'] = np.array([i[1] for i in numbers])
mtsi['sigma'] = np.array([i[2] for i in numbers])
mtsi['n'] = np.arange(len(numbers))
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

mu_list = np.unique(mts['fbmp']['mu'])
mu_std_tru_list = np.sqrt(mu_list)

mts = {'lucyddm':mtsi.copy(), 'takara':mtsi.copy(), 'xiaopeip':mtsi.copy(), 'fbmp':mtsi.copy()}
# mts = {'lucyddm':mtsi.copy(), 'fbmp':mtsi.copy()}

use_log = False
for key in tqdm(mts.keys()):
    for i in range(len(mts[key])):
        f = filelist[mts[key][i]['n']]
        mu = mts[key][i]['mu']
        tau = mts[key][i]['tau']
        sigma = mts[key][i]['sigma']

        try:
            with h5py.File(os.path.join('result', key, 'solu', f), 'r', libver='latest', swmr=True) as soluf, h5py.File(os.path.join('result', key, 'dist', f), 'r', libver='latest', swmr=True) as distf, h5py.File(os.path.join('waveform', f), 'r', libver='latest', swmr=True) as wavef:
                time = soluf['starttime'][:]
                starttime_attrs = dict(soluf['starttime'].attrs)
                start = wavef['SimTruth/T'][:]
                pelist = wavef['SimTriggerInfo/PEList'][:]
                waves = wavef['Readout/Waveform'][:]
                gmu = wavef['SimTriggerInfo/PEList'].attrs['gmu']
                gsigma = wavef['SimTriggerInfo/PEList'].attrs['gsigma']
                r = wavef['SimTruth/T'].attrs['r']
                r = np.inf
            mts[key][i]['N'] = len(start)
            vali = np.abs(time['tscharge'] - start['T0'] - np.mean(time['tscharge'] - start['T0'])) <= r * np.std(time['tscharge'] - start['T0'], ddof=-1)
            # vali = np.full(len(time), True)
            Chnum = len(np.unique(pelist['PMTId']))
            e_ans, i_ans = np.unique(pelist['TriggerNo'] * Chnum + pelist['PMTId'], return_index=True)
            i_ans = np.append(i_ans, len(pelist))
            pe_sum = np.array([pelist[i_ans[i]:i_ans[i+1]]['Charge'].sum() for i in range(len(e_ans))]) / gmu
            wave_sum = waves['Waveform'].sum(axis=1) / gmu

            # n = np.arange(1, 1000)
            # mean = np.average(n, weights=poisson.pmf(n, mu=mu))
            # lognm = np.average(np.log(n), weights=poisson.pmf(n, mu=mu))
            # s = np.sqrt(np.average((np.log(n) - mean)**2, weights=poisson.pmf(n, mu=mu)))

            npe = np.diff(i_ans)
            N_add = N / (1 - poisson.cdf(0, mu)) - N
            # s_npe = np.std(npe, ddof=-1)
            s_npe = np.sqrt(mu)
            s_wave_sum = np.std(np.append(wave_sum[vali], np.zeros(round(N_add))), ddof=-1)
            bias_wave_sum = np.mean(np.append(wave_sum[vali], np.zeros(round(N_add)))) - mu
            s_pe_sum = np.std(np.append(pe_sum[vali], np.zeros(round(N_add))), ddof=-1)
            bias_pe_sum = np.mean(np.append(pe_sum[vali], np.zeros(round(N_add)))) - mu
            # Use sample STD to estimate STD
            s_mucharge = np.std(time['mucharge'][vali], ddof=-1)
            bias_mucharge = np.mean(time['mucharge'][vali]) - mu
            s_muwave = np.std(np.append(time['muwave'][vali], np.zeros(round(N_add))), ddof=-1)
            bias_muwave = np.mean(np.append(time['muwave'][vali], np.zeros(round(N_add)))) - mu
            # Global fit STD
            # s_mucharge = np.nan
            # bias_mucharge = np.nan
            # s_muwave = starttime_attrs['sigmamu']
            # bias_muwave = starttime_attrs['mu'] - mu

            mts[key][i]['stdmutru'] = s_npe
            mts[key][i]['meanmutru'] = mu
            mts[key][i]['stdmuint'] = s_wave_sum
            mts[key][i]['biasmuint'] = bias_wave_sum
            mts[key][i]['stdmupe'] = s_pe_sum
            mts[key][i]['biasmupe'] = bias_pe_sum
            mts[key][i]['stdmumax'] = s_mucharge
            mts[key][i]['biasmumax'] = bias_mucharge
            mts[key][i]['stdmu'] = s_muwave
            mts[key][i]['biasmu'] = bias_muwave
        except:
            pass

keylist = mts.keys()
figd = plt.figure(figsize=(len(Tau) * 5, len(Sigma) * 3))
gs = gridspec.GridSpec(1, 2, figure=figdd, left=0.1, right=0.8, top=0.92, bottom=0.15, wspace=0.3, hspace=0.35)
figdd = plt.figure(figsize=(len(Tau) * 5, len(Sigma) * 3))
gs = gridspec.GridSpec(1, 2, figure=figdd, left=0.1, right=0.8, top=0.92, bottom=0.15, wspace=0.3, hspace=0.35)
figbr = plt.figure(figsize=(len(Tau) * 5, len(Sigma) * 3))
gs = gridspec.GridSpec(1, 2, figure=figbr, left=0.1, right=0.8, top=0.92, bottom=0.15, wspace=0.3, hspace=0.35)
figb = plt.figure(figsize=(len(Tau) * 5, len(Sigma) * 3))
gs = gridspec.GridSpec(1, 2, figure=figb, left=0.1, right=0.8, top=0.92, bottom=0.15, wspace=0.3, hspace=0.35)
for i, sigma in enumerate(Sigma):
    for j, tau in enumerate(Tau):
        # charge std
        stdlist = mts['fbmp'][(mts['fbmp']['tau'] == tau) & (mts['fbmp']['sigma'] == sigma)]
        ax = figd.add_subplot(gs[i, j])
        yerr = np.vstack([stdlist['stdmuint']-np.sqrt(np.power(stdlist['stdmuint'],2)*stdlist['N']/chi2.ppf(1-alpha/2, stdlist['N'])), np.sqrt(np.power(stdlist['stdmuint'],2)*stdlist['N']/chi2.ppf(alpha/2, stdlist['N']))-stdlist['stdmuint']]) / (stdlist['biasmuint'] + stdlist['meanmutru'])
        ax.errorbar(stdlist['mu'] + jitter['tru'], stdlist['stdmuint'] / (stdlist['biasmuint'] + stdlist['meanmutru']) / (1 / np.sqrt(stdlist['mu'])), yerr=yerr / (1 / np.sqrt(stdlist['mu'])), label='$\mathrm{int}$', c=color['1st'], marker=marker['1st'])
        for key in keylist:
            stdlist = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
            yerr = np.vstack([stdlist['stdmu']-np.sqrt(np.power(stdlist['stdmu'],2)*stdlist['N']/chi2.ppf(1-alpha/2, stdlist['N'])), np.sqrt(np.power(stdlist['stdmu'],2)*stdlist['N']/chi2.ppf(alpha/2, stdlist['N']))-stdlist['stdmu']]) / (stdlist['biasmu'] + stdlist['meanmutru'])
            ax.errorbar(stdlist['mu'] + jitter[key], stdlist['stdmu'] / (stdlist['biasmu'] + stdlist['meanmutru']) / (1 / np.sqrt(stdlist['mu'])), yerr=yerr / (1 / np.sqrt(stdlist['mu'])), label='$' + label[key] + '$', c=color[key], marker=marker[key])
        ax.axhline(y=1, color='k', alpha=0.5)
        ax.set_xlabel(r'$\mu$')
        ax.set_ylabel(r'$\mathrm{ratio}$')
        ax.set_title(fr'$\tau_l={tau}\si{{ns}},\,\sigma_l={sigma}\si{{ns}}$')
        # ax.set_ylim(0.96, 1.54)
        ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
        ax.grid()
        if i == len(Sigma) - 1 and j == len(Tau) - 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1., 0.9))

        stdlist = mts['fbmp'][(mts['fbmp']['tau'] == tau) & (mts['fbmp']['sigma'] == sigma)]
        ax = figdd.add_subplot(gs[i, j])
        yerr = np.vstack([stdlist['stdmuint']-np.sqrt(np.power(stdlist['stdmuint'],2)*stdlist['N']/chi2.ppf(1-alpha/2, stdlist['N'])), np.sqrt(np.power(stdlist['stdmuint'],2)*stdlist['N']/chi2.ppf(alpha/2, stdlist['N']))-stdlist['stdmuint']]) / (stdlist['biasmuint'] + stdlist['meanmutru'])
        ax.errorbar(stdlist['mu'] + jitter['tru'], stdlist['stdmuint'] / (stdlist['biasmuint'] + stdlist['meanmutru']), yerr=yerr, label='$\mathrm{int}$', c=color['1st'], marker=marker['1st'])
        for key in keylist:
            stdlist = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
            yerr = np.vstack([stdlist['stdmu']-np.sqrt(np.power(stdlist['stdmu'],2)*stdlist['N']/chi2.ppf(1-alpha/2, stdlist['N'])), np.sqrt(np.power(stdlist['stdmu'],2)*stdlist['N']/chi2.ppf(alpha/2, stdlist['N']))-stdlist['stdmu']]) / (stdlist['biasmu'] + stdlist['meanmutru'])
            ax.errorbar(stdlist['mu'] + jitter[key], stdlist['stdmu'] / (stdlist['biasmu'] + stdlist['meanmutru']), yerr=yerr, label='$' + label[key] + '$', c=color[key], marker=marker[key])
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
        
        # charge bias ratio
        stdlist = mts['fbmp'][(mts['fbmp']['tau'] == tau) & (mts['fbmp']['sigma'] == sigma)]
        ax = figbr.add_subplot(gs[i, j])
        # yerr = stdlist['biasmuint'] / np.sqrt(stdlist['N']) / stdlist['meanmutru']
        yerr = np.vstack([stdlist['biasmuint']-t.ppf(1-alpha/2, stdlist['N'])*stdlist['stdmuint']/np.sqrt(stdlist['N']), t.ppf(1-alpha/2, stdlist['N'])*stdlist['stdmuint']/np.sqrt(stdlist['N'])-stdlist['biasmuint']]) / stdlist['meanmutru']
        ax.errorbar(stdlist['mu'] + jitter['tru'], stdlist['biasmuint'] / stdlist['meanmutru'], yerr=yerr, label='$\mathrm{int}$', c=color['1st'], marker=marker['1st'])
        # ax.plot(stdlist['mu'] + jitter['tru'], stdlist['biasmuint'] / stdlist['meanmutru'], label='$\mathrm{int}$', c=color['1st'], marker=marker['1st'])
        for key in keylist:
            stdlist = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
            # if key == 'fbmp':
            #     yerr = stdlist['biasmumax'] / np.sqrt(stdlist['N']) / stdlist['meanmutru']
            #     ax.errorbar(stdlist['mu'], stdlist['biasmumax'] / stdlist['meanmutru'], yerr=yerr, label='$' + label['fbmpone'] + '$', c=color['fbmpone'], marker=marker['fbmpone'])
            yerr = np.vstack([stdlist['biasmu']-t.ppf(1-alpha/2, stdlist['N'])*stdlist['stdmu']/np.sqrt(stdlist['N']), t.ppf(1-alpha/2, stdlist['N'])*stdlist['stdmu']/np.sqrt(stdlist['N'])-stdlist['biasmu']]) / stdlist['meanmutru']
            ax.errorbar(stdlist['mu'] + jitter[key], stdlist['biasmu'] / stdlist['meanmutru'], yerr=yerr, label='$' + label[key] + '$', c=color[key], marker=marker[key])
            # ax.plot(stdlist['mu'] + jitter[key], stdlist['biasmu'] / stdlist['meanmutru'], label='$' + label[key] + '$', c=color[key], marker=marker[key])
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

        # charge bias
        stdlist = mts['fbmp'][(mts['fbmp']['tau'] == tau) & (mts['fbmp']['sigma'] == sigma)]
        ax = figb.add_subplot(gs[i, j])
        # yerr = stdlist['biasmuint'] / np.sqrt(stdlist['N'])
        yerr = np.vstack([stdlist['biasmuint']-t.ppf(1-alpha/2, stdlist['N'])*stdlist['stdmuint']/np.sqrt(stdlist['N']), t.ppf(1-alpha/2, stdlist['N'])*stdlist['stdmuint']/np.sqrt(stdlist['N'])-stdlist['biasmuint']])
        ax.errorbar(stdlist['mu'] + jitter['tru'], stdlist['biasmuint'], yerr=yerr, label='$\mathrm{int}$', c=color['1st'], marker=marker['1st'])
        # ax.plot(stdlist['mu'] + jitter['tru'], stdlist['biasmuint'], label='$\mathrm{int}$', c=color['1st'], marker=marker['1st'])
        for key in keylist:
            stdlist = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
            # if key == 'fbmp':
            #     yerr = stdlist['biasmumax'] / np.sqrt(stdlist['N'])
            #     ax.errorbar(stdlist['mu'], stdlist['biasmumax'], yerr=yerr, label='$' + label['fbmpone'] + '$', c=color['fbmpone'], marker=marker['fbmpone'])
            yerr = np.vstack([stdlist['biasmu']-t.ppf(1-alpha/2, stdlist['N'])*stdlist['stdmu']/np.sqrt(stdlist['N']), t.ppf(1-alpha/2, stdlist['N'])*stdlist['stdmu']/np.sqrt(stdlist['N'])-stdlist['biasmu']])
            ax.errorbar(stdlist['mu'] + jitter[key], stdlist['biasmu'], yerr=yerr, label='$' + label[key] + '$', c=color[key], marker=marker[key])
            # ax.plot(stdlist['mu'] + jitter[key], stdlist['biasmu'], label='$' + label[key] + '$', c=color[key], marker=marker[key])
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
figd.savefig('Note/figures/vs-deltamethodsdivmu.pgf')
figd.savefig('Note/figures/vs-deltamethodsdivmu.pdf')
figd.savefig('Note/figures/vs-deltamethodsdivmu.png')
plt.close(figd)
figdd.savefig('Note/figures/vs-deltamu.pgf')
figdd.savefig('Note/figures/vs-deltamu.pdf')
figdd.savefig('Note/figures/vs-deltamu.png')
plt.close(figdd)
figbr.savefig('Note/figures/vs-biasmu.pgf')
figbr.savefig('Note/figures/vs-biasmu.pdf')
figbr.savefig('Note/figures/vs-biasmu.png')
plt.close(figbr)
figb.savefig('Note/figures/vs-biasmut.pgf')
figb.savefig('Note/figures/vs-biasmut.pdf')
figb.savefig('Note/figures/vs-biasmut.png')
plt.close(figb)
