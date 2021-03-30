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
from scipy.stats import poisson, uniform, chi2, t
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

filelist = os.listdir('result/lucyddm/solu')
filelist = [f for f in filelist if f[0] != '.' and os.path.splitext(f)[-1] == '.h5']
numbers = [[float(i) for i in f[:-3].split('-')] for f in filelist]
stype = np.dtype([('mu', np.float64), ('tau', np.float64), ('sigma', np.float64), ('n', np.uint), ('std1sttruth', np.float64), ('stdtruth', np.float64), ('stdcharge', np.float64), ('stdwave', np.float64), ('bias1sttruth', np.float64), ('biastruth', np.float64), ('biascharge', np.float64), ('biaswave', np.float64), ('wdist', np.float64, 3), ('RSS', np.float64, 3), ('N', np.uint), ('stdchargesuccess', np.uint), ('stdwavesuccess', np.uint)])
mtsi = np.zeros(len(numbers), dtype=stype)
mtsi['mu'] = np.array([i[0] for i in numbers])
mtsi['tau'] = np.array([i[1] for i in numbers])
mtsi['sigma'] = np.array([i[2] for i in numbers])
mtsi['n'] = np.arange(len(numbers))
mtsi['std1sttruth'] = np.nan
mtsi['stdtruth'] = np.nan
mtsi['stdcharge'] = np.nan
mtsi['stdwave'] = np.nan
mtsi['bias1sttruth'] = np.nan
mtsi['biastruth'] = np.nan
mtsi['biascharge'] = np.nan
mtsi['biaswave'] = np.nan
mtsi['wdist'] = np.nan
mtsi['RSS'] = np.nan
mtsi = np.sort(mtsi, kind='stable', order=['mu', 'tau', 'sigma'])

mts = {'findpeak':mtsi.copy(), 'threshold':mtsi.copy(), 'fftrans':mtsi.copy(), 'lucyddm':mtsi.copy(), 'xiaopeip':mtsi.copy(), 'mcmc':mtsi.copy(), 'takara':mtsi.copy(), 'fbmp':mtsi.copy()}
deltalabel = {'1st':'\mathrm{1st}', 'tru':'\mathrm{ALL}', 'findpeak':'\mathrm{FindPeak}', 'threshold':'\mathrm{Shift}', 'fftrans':'\mathrm{FFT}', 'lucyddm':'\mathrm{LucyDDM}', 'xiaopeip':'\mathrm{Fitting}', 'takara':'\mathrm{CNN}', 'fbmp':'\mathrm{FBMPcha}', 'fbmpwave':'\mathrm{FBMPt0}', 'mcmc':'\mathrm{MCMCcha}', 'mcmcwave':'\mathrm{MCMCt0}'}
label = {'1st':'\mathrm{1st}', 'tru':'\mathrm{ALL}', 'findpeak':'\mathrm{FindPeak}', 'threshold':'\mathrm{Shift}', 'fftrans':'\mathrm{FFT}', 'lucyddm':'\mathrm{LucyDDM}', 'xiaopeip':'\mathrm{Fitting}', 'takara':'\mathrm{CNN}', 'fbmp':'\mathrm{FBMP}', 'fbmpwave':'\mathrm{FBMP}', 'mcmc':'\mathrm{MCMC}', 'mcmcwave':'\mathrm{MCMC}'}
marker = {'1st':'s', 'tru':'h', 'findpeak':',', 'threshold':'1', 'fftrans':'+', 'lucyddm':'p', 'xiaopeip':'*', 'takara':'X', 'fbmp':'o', 'fbmpwave':'^', 'mcmc':'x', 'mcmcwave':'>'}
color = {'1st':'b', 'tru':'k', 'findpeak':'C1', 'threshold':'C2', 'fftrans':'m', 'lucyddm':'y', 'xiaopeip':'c', 'takara':'C0', 'fbmp':'r', 'fbmpwave':'b', 'mcmc':'C4', 'mcmcwave':'C5'}
jit = 0.05
jitter = {'mcmc':-6 * jit, 'tru':-5 * jit, 'fbmp':-4 * jit, 'fbmpwave':-3 * jit, 'lucyddm':-2 * jit, 'mcmcwave':-1 * jit, 'xiaopeip':1 * jit, 'takara':0 * jit, 'fftrans':2 * jit, '1st':3 * jit, 'findpeak':4 * jit, 'threshold':5 * jit}

for key in mts.keys():
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
                r = wavef['SimTruth/T'].attrs['r'] * 1.5
            vali = np.abs(time['tscharge'] if time['tscharge'].ndim == 1 else time['tscharge'][:, 1] - start['T0'] - np.mean(time['tscharge'] if time['tscharge'].ndim == 1 else time['tscharge'][:, 1] - start['T0'])) < r * np.std(time['tscharge'] if time['tscharge'].ndim == 1 else time['tscharge'][:, 1] - start['T0'], ddof=-1)
            mts[key][i]['N'] = len(start)
            mts[key][i]['std1sttruth'] = np.std(start['ts1sttruth'] - start['T0'], ddof=-1)
            mts[key][i]['stdtruth'] = np.std(start['tstruth'] - start['T0'], ddof=-1)
            mts[key][i]['stdcharge'], mts[key][i]['stdchargesuccess'] = wff.stdrmoutlier(time['tscharge'] - start['T0'] if time['tscharge'].ndim == 1 else time['tscharge'][:, 1] - start['T0'], r)
            mts[key][i]['bias1sttruth'] = np.mean(start['ts1sttruth'] - start['T0'])
            mts[key][i]['biastruth'] = np.mean(start['tstruth'] - start['T0'])
            mts[key][i]['biascharge'] = np.mean((time['tscharge'] if time['tscharge'].ndim == 1 else time['tscharge'][:, 1])[vali] - start['T0'][vali])
            if not np.any(np.isnan(time['tswave'][vali])):
                mts[key][i]['stdwave'], mts[key][i]['stdwavesuccess'] = wff.stdrmoutlier(time['tswave'] - start['T0'] if time['tscharge'].ndim == 1 else time['tswave'][:, 1] - start['T0'], r)
                mts[key][i]['biaswave'] = np.mean((time['tswave'] if time['tscharge'].ndim == 1 else time['tswave'][:, 1])[vali] - start['T0'][vali])
            mts[key][i]['wdist'] = np.insert(np.percentile(record['wdist'][vali], [5, 95]), 1, record['wdist'][vali].mean())
            mts[key][i]['RSS'] = np.insert(np.percentile(record['RSS'][vali], [5, 95]), 1, record['RSS'][vali].mean())
        except:
            pass

dhigh = np.array([[np.max(mts[key]['std1sttruth']), np.max(mts[key]['stdtruth']), np.max(mts[key]['stdcharge']), np.max(mts[key]['stdwave'])] for key in mts.keys()])
dhigh = np.max(dhigh[~np.isnan(dhigh)]) * 1.05
whigh = np.array([[np.max(mts[key]['wdist'])] for key in mts.keys()])
whigh = np.max(whigh[~np.isnan(whigh)]) * 1.05
rhigh = np.array([[np.max(mts[key]['RSS'])] for key in mts.keys()])
rhigh = np.max(rhigh[~np.isnan(rhigh)]) * 1.05

figd = plt.figure(figsize=(len(Tau) * 5, len(Sigma) * 4.5))
gsd = gridspec.GridSpec(len(Sigma), len(Tau), figure=figd, left=0.1, right=0.8, top=0.93, bottom=0.1, wspace=0.3, hspace=0.35)
figb = plt.figure(figsize=(len(Tau) * 5, len(Sigma) * 4.5))
gsb = gridspec.GridSpec(len(Sigma), len(Tau), figure=figd, left=0.1, right=0.8, top=0.93, bottom=0.1, wspace=0.3, hspace=0.35)
figdd = plt.figure(figsize=(len(Tau) * 5, len(Sigma) * 4.5))
gsdd = gridspec.GridSpec(len(Sigma), len(Tau), figure=figd, left=0.1, right=0.8, top=0.93, bottom=0.1, wspace=0.3, hspace=0.35)
figw = plt.figure(figsize=(len(Tau) * 5, len(Sigma) * 4.5))
gsw = gridspec.GridSpec(len(Sigma), len(Tau), figure=figw, left=0.1, right=0.8, top=0.93, bottom=0.1, wspace=0.3, hspace=0.35)
figr = plt.figure(figsize=(len(Tau) * 5, len(Sigma) * 4.5))
gsr = gridspec.GridSpec(len(Sigma), len(Tau), figure=figr, left=0.1, right=0.8, top=0.93, bottom=0.1, wspace=0.3, hspace=0.35)
alpha = 0.05
lim = {'deltadiv':np.tile([0.3, 0.5, 0.], (2, 1)), 'wdist':np.tile([2, 3.5, 7], (2, 1)), 'rss':np.array([[0.7e3, 2.5e3, 2e3], [1.5e3, 2.5e3, 2e3]])}
keylist = list(mts.keys())
for sigma, i in zip(Sigma, list(range(len(Sigma)))):
    for tau, j in zip(Tau, list(range(len(Tau)))):
        ax = figd.add_subplot(gsd[i, j])
        stdlist = mts['lucyddm'][(mts['lucyddm']['tau'] == tau) & (mts['lucyddm']['sigma'] == sigma)]
        yerr1st = np.vstack([stdlist['std1sttruth']-np.sqrt(np.power(stdlist['std1sttruth'],2)*stdlist['N']/chi2.ppf(1-alpha/2, stdlist['N'])), np.sqrt(np.power(stdlist['std1sttruth'],2)*stdlist['N']/chi2.ppf(alpha/2, stdlist['N']))-stdlist['std1sttruth']])
        yerrall = np.vstack([stdlist['stdtruth']-np.sqrt(np.power(stdlist['stdtruth'],2)*stdlist['N']/chi2.ppf(1-alpha/2, stdlist['N'])), np.sqrt(np.power(stdlist['stdtruth'],2)*stdlist['N']/chi2.ppf(alpha/2, stdlist['N']))-stdlist['stdtruth']])
        ax.errorbar(stdlist['mu'] + jitter['1st'], stdlist['std1sttruth'], yerr=yerr1st, c=color['1st'], label='$\delta_'+deltalabel['1st']+'$', marker=marker['1st'])
        ax.errorbar(stdlist['mu'] + jitter['tru'], stdlist['stdtruth'], yerr=yerrall, c=color['tru'], label='$\delta_'+deltalabel['tru']+'$', marker=marker['tru'])
        for m in ['mcmc', 'fbmp']:
            stdlistwav = mts[m][(mts[m]['tau'] == tau) & (mts[m]['sigma'] == sigma)]
            yerrwav = np.vstack([stdlistwav['stdwave']-np.sqrt(np.power(stdlistwav['stdwave'],2)*stdlistwav['N']/chi2.ppf(1-alpha/2, stdlistwav['N'])), np.sqrt(np.power(stdlistwav['stdwave'],2)*stdlistwav['N']/chi2.ppf(alpha/2, stdlistwav['N']))-stdlistwav['stdwave']])
            ax.errorbar(stdlistwav['mu'] + jitter[m + 'wave'], stdlistwav['stdwave'], yerr=yerrwav, c=color[m + 'wave'], label='$\delta_'+deltalabel[m + 'wave']+'$', marker=marker[m + 'wave'])
        for k in range(len(keylist)):
            key = keylist[k]
            stdlist = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
            yerrcha = np.vstack([stdlist['stdcharge']-np.sqrt(np.power(stdlist['stdcharge'],2)*stdlist['N']/chi2.ppf(1-alpha/2, stdlist['N'])), np.sqrt(np.power(stdlist['stdcharge'],2)*stdlist['N']/chi2.ppf(alpha/2, stdlist['N']))-stdlist['stdcharge']])
            ax.errorbar(stdlist['mu'] + jitter[key], stdlist['stdcharge'], yerr=yerrcha, c=color[key], label='$\delta_'+deltalabel[key]+'$', marker=marker[key])
        ax.set_xlabel(r'$N_{\mathrm{PE}}\ \mathrm{expectation}\ \mu$')
        ax.set_ylabel(r'$\delta/\si{ns}$')
        ax.set_title(fr'$\tau={tau}\si{{ns}},\,\sigma={sigma}\si{{ns}}$')
        # ax.set_yscale('log')
        ax.grid()
        if i == len(Sigma) - 1 and j == len(Tau) - 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1., 0.9))

        ax = figb.add_subplot(gsb[i, j])
        stdlist = mts['lucyddm'][(mts['lucyddm']['tau'] == tau) & (mts['lucyddm']['sigma'] == sigma)]
        yerr1st = np.vstack([stdlist['bias1sttruth']-t.ppf(1-alpha/2, stdlist['N'])*stdlist['std1sttruth']/np.sqrt(stdlist['N']), t.ppf(1-alpha/2, stdlist['N'])*stdlist['std1sttruth']/np.sqrt(stdlist['N'])-stdlist['bias1sttruth']])
        yerrall = np.vstack([stdlist['biastruth']-t.ppf(1-alpha/2, stdlist['N'])*stdlist['stdtruth']/np.sqrt(stdlist['N']), t.ppf(1-alpha/2, stdlist['N'])*stdlist['stdtruth']/np.sqrt(stdlist['N'])-stdlist['biastruth']])
        ax.errorbar(stdlist['mu'] + jitter['1st'], stdlist['bias1sttruth'], yerr=yerr1st, c=color['1st'], label='$\mathrm{bias}_'+deltalabel['1st']+'$', marker=marker['1st'])
        ax.errorbar(stdlist['mu'] + jitter['tru'], stdlist['biastruth'], yerr=yerrall, c=color['tru'], label='$\mathrm{bias}_'+deltalabel['tru']+'$', marker=marker['tru'])
        for m in ['mcmc', 'fbmp']:
            stdlistwav = mts[m][(mts[m]['tau'] == tau) & (mts[m]['sigma'] == sigma)]
            yerrwav = np.vstack([stdlistwav['biaswave']-t.ppf(1-alpha/2, stdlist['N'])*stdlist['stdwave']/np.sqrt(stdlist['N']), t.ppf(1-alpha/2, stdlist['N'])*stdlist['stdwave']/np.sqrt(stdlist['N'])-stdlistwav['biaswave']])
            ax.errorbar(stdlistwav['mu'] + jitter[m + 'wave'], stdlistwav['biaswave'], yerr=yerrwav, c=color[m + 'wave'], label='$\mathrm{bias}_'+deltalabel[m + 'wave']+'$', marker=marker[m + 'wave'])
        for k in range(len(keylist)):
            key = keylist[k]
            stdlist = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
            yerrcha = np.vstack([stdlist['biascharge']-t.ppf(1-alpha/2, stdlist['N'])*stdlist['stdcharge']/np.sqrt(stdlist['N']), t.ppf(1-alpha/2, stdlist['N'])*stdlist['stdcharge']/np.sqrt(stdlist['N'])-stdlist['biascharge']])
            ax.errorbar(stdlist['mu'] + jitter[key], stdlist['biascharge'], yerr=yerrcha, c=color[key], label='$\mathrm{bias}_'+deltalabel[key]+'$', marker=marker[key])
        ax.set_xlabel(r'$N_{\mathrm{PE}}\ \mathrm{expectation}\ \mu$')
        ax.set_ylabel(r'$\mathrm{bias}/\si{ns}$')
        ax.set_title(fr'$\tau={tau}\si{{ns}},\,\sigma={sigma}\si{{ns}}$')
        # ax.set_yscale('log')
        ax.grid()
        if i == len(Sigma) - 1 and j == len(Tau) - 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1., 0.9))

        ax = figdd.add_subplot(gsdd[i, j])
        stdlist = mts['lucyddm'][(mts['lucyddm']['tau'] == tau) & (mts['lucyddm']['sigma'] == sigma)]
        yerrall = stdlist['stdtruth'] / stdlist['std1sttruth'] / np.sqrt(stdlist['N'])
        ax.errorbar(stdlist['mu'] + jitter['tru'], stdlist['stdtruth'] / stdlist['std1sttruth'], yerr=yerrall, label='$\delta_'+deltalabel['tru']+'/\delta_'+deltalabel['1st']+'$', c=color['tru'], marker=marker['tru'])
        # ax.plot(stdlist['mu'] + jitter['tru'], stdlist['stdtruth'] / stdlist['std1sttruth'], label='$\delta_'+deltalabel['tru']+'/\delta_'+deltalabel['1st']+'$', c=color['tru'], marker=marker['tru'])
        for k in range(len(keylist)):
            key = keylist[k]
            if key == 'findpeak' or key == 'fftrans':
                continue
            stdlistkey = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
            yerr = stdlistkey['stdcharge'] / stdlist['std1sttruth'] / np.sqrt(stdlistkey['N'])
            ax.errorbar(stdlist['mu'] + jitter[key], stdlistkey['stdcharge'] / stdlist['std1sttruth'], yerr=yerr, label='$\delta_'+deltalabel[key]+'/\delta_'+deltalabel['1st']+'$', c=color[key], marker=marker[key])
            # ax.plot(stdlist['mu'] + jitter[key], stdlistkey['stdcharge'] / stdlist['std1sttruth'], label='$\delta_'+deltalabel[key]+'/\delta_'+deltalabel['1st']+'$', c=color[key], marker=marker[key])
        for m in ['mcmc', 'fbmp']:
            stdlistwav = mts[m][(mts[m]['tau'] == tau) & (mts[m]['sigma'] == sigma)]
            yerrwav = stdlistwav['stdcharge'] / stdlist['std1sttruth'] / np.sqrt(stdlistwav['N'])
            ax.errorbar(stdlistwav['mu'] + jitter[m + 'wave'], stdlistwav['stdwave'] / stdlist['std1sttruth'], yerr=yerrwav, label='$\delta_'+deltalabel[m + 'wave']+'/\delta_'+deltalabel['1st']+'$', c=color[m + 'wave'], marker=marker[m + 'wave'])
            # ax.plot(stdlistwav['mu'] + jitter[m + 'wave'], stdlistwav['stdwave'] / stdlist['std1sttruth'], label='$\delta_'+deltalabel[m + 'wave']+'/\delta_'+deltalabel['1st']+'$', c=color[m + 'wave'], marker=marker[m + 'wave'])
        ax.set_xlabel(r'$N_{\mathrm{PE}}\ \mathrm{expectation}\ \mu$')
        ax.set_ylabel(r'$ratio$')
        ax.set_title(fr'$\tau={tau}\si{{ns}},\,\sigma={sigma}\si{{ns}}$')
        ax.set_ylim(lim['deltadiv'][i, j], 1.01)
        ax.grid()
        if i == len(Sigma) - 1 and j == len(Tau) - 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1., 0.9))

        ax = figw.add_subplot(gsw[i, j])
        for k in range(len(keylist)):
            key = keylist[k]
            if key == 'findpeak' or key == 'threshold' or key == 'fftrans':
                continue
            wdistlist = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
            # ax.plot(wdistlist['mu'] + jitter[key], wdistlist['wdist'], c=color[key], label='$'+label[key]+'$', marker=marker[key])
            yerr = np.vstack([wdistlist['wdist'][:, 1] - wdistlist['wdist'][:, 0], wdistlist['wdist'][:, 2] - wdistlist['wdist'][:, 1]])
            ax.errorbar(wdistlist['mu'] + jitter[key], wdistlist['wdist'][:, 1], yerr=yerr, label='$'+label[key]+'$', c=color[key], marker=marker[key])
            ax.fill_between(wdistlist['mu'] + jitter[key], wdistlist['wdist'][:, 0], wdistlist['wdist'][:, 2], fc=color[key], alpha=0.1, color=None)
        ax.set_xlabel(r'$N_{\mathrm{PE}}\ \mathrm{expectation}\ \mu$')
        ax.set_ylabel(r'$W-dist/\si{ns}$')
        ax.set_title(fr'$\tau={tau}\si{{ns}},\,\sigma={sigma}\si{{ns}}$')
        ax.set_ylim(0, lim['wdist'][i, j])
        # ax.set_yscale('log')
        ax.grid()
        if i == len(Sigma) - 1 and j == len(Tau) - 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1., 0.9))

        ax = figr.add_subplot(gsr[i, j])
        for k in range(len(keylist)):
            key = keylist[k]
            if key == 'findpeak' or key == 'threshold' or key == 'fftrans':
                continue
            rsslist = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
            # ax.plot(rsslist['mu'] + jitter[key], rsslist['RSS'], c=color[key], label='$'+label[key]+'$', marker=marker[key])
            yerr = np.vstack([rsslist['RSS'][:, 1] - rsslist['RSS'][:, 0], rsslist['RSS'][:, 2] - rsslist['RSS'][:, 1]])
            ax.errorbar(rsslist['mu'] + jitter[key], rsslist['RSS'][:, 1], yerr=yerr, label='$'+label[key]+'$', c=color[key], marker=marker[key])
            ax.fill_between(rsslist['mu'] + jitter[key], rsslist['RSS'][:, 0], rsslist['RSS'][:, 2], fc=color[key], alpha=0.1, color=None)
        ax.set_xlabel(r'$N_{\mathrm{PE}}\ \mathrm{expectation}\ \mu$')
        ax.set_ylabel(r'$RSS/\si{mV}^{2}$')
        ax.set_title(fr'$\tau={tau}\si{{ns}},\,\sigma={sigma}\si{{ns}}$')
        ax.set_ylim(0, lim['rss'][i, j])
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

alpha = 0.05
marker = [['s', '^']]
colors = [['r', 'b']]
fig = plt.figure(figsize=(8, 4))
gs = gridspec.GridSpec(1, 2, figure=fig, left=0.1, right=0.85, top=0.93, bottom=0.15, wspace=0.25, hspace=0.2)
ax = fig.add_subplot(gs[0, 0])
std1sttruth = np.empty(len(stdlist['mu']))
stdlist = mts['lucyddm'][(mts['lucyddm']['tau'] == Tau[0]) & (mts['lucyddm']['sigma'] == Sigma[0])]
sigma = 0
tau = Tau[1]
np.random.seed(0)
for mu, i in zip(stdlist['mu'], list(range(len(stdlist['mu'])))):
    N = stdlist['N'][i]
    npe = poisson.ppf(1 - uniform.rvs(scale=1-poisson.cdf(0, mu), size=N), mu).astype(int)
    t0 = np.random.uniform(100., 500., size=N)
    sams = [wff.time(npe[j], tau, sigma) + t0[j] for j in range(N)]
    ts1sttruth = np.array([np.min(sams[j]) for j in range(N)])
    std1sttruth[i] = np.std(ts1sttruth - t0, ddof=-1)
yerr = np.vstack([std1sttruth-np.sqrt(np.power(std1sttruth,2)*stdlist['N']/chi2.ppf(1-alpha/2, stdlist['N'])), np.sqrt(np.power(std1sttruth,2)*stdlist['N']/chi2.ppf(alpha/2, stdlist['N']))-std1sttruth])
ax.errorbar(stdlist['mu'], std1sttruth, yerr=yerr, label='$\delta_'+deltalabel['tru']+fr',\,\tau={tau}\si{{ns}},\,\sigma={sigma}\si{{ns}}$', marker='o', color='g')
ax.errorbar(stdlist['mu'], std1sttruth, yerr=yerr, label='$\delta_'+deltalabel['1st']+fr',\,\tau={tau}\si{{ns}},\,\sigma={sigma}\si{{ns}}$', marker='o', color='g', linestyle='dashed')
for sigma, i in zip(Sigma, list(range(len(Sigma)))):
    for tau, j in zip(Tau, list(range(len(Tau)))):
        stdlist = mts['lucyddm'][(mts['lucyddm']['tau'] == tau) & (mts['lucyddm']['sigma'] == sigma)]
        yerrall = np.vstack([stdlist['stdtruth']-np.sqrt(np.power(stdlist['stdtruth'],2)*stdlist['N']/chi2.ppf(1-alpha/2, stdlist['N'])), np.sqrt(np.power(stdlist['stdtruth'],2)*stdlist['N']/chi2.ppf(alpha/2, stdlist['N']))-stdlist['stdtruth']])
        yerr1st = np.vstack([stdlist['std1sttruth']-np.sqrt(np.power(stdlist['std1sttruth'],2)*stdlist['N']/chi2.ppf(1-alpha/2, stdlist['N'])), np.sqrt(np.power(stdlist['std1sttruth'],2)*stdlist['N']/chi2.ppf(alpha/2, stdlist['N']))-stdlist['std1sttruth']])
        ax.errorbar(stdlist['mu'], stdlist['stdtruth'], yerr=yerrall, marker=marker[i][j], color=colors[i][j])
        ax.errorbar(stdlist['mu'], stdlist['std1sttruth'], yerr=yerr1st, marker=marker[i][j], color=colors[i][j], linestyle='dashed')
ax.set_xlabel(r'$N_{\mathrm{PE}}\ \mathrm{expectation}\ \mu$')
ax.set_ylabel(r'$\mathrm{timing\ resolution}/\si{ns}$')
ax.grid()
# ax.legend(loc='upper right')
ax = fig.add_subplot(gs[0, 1])
sigma = 0
tau = Tau[1]
stdlist = mts['lucyddm'][(mts['lucyddm']['tau'] == Tau[0]) & (mts['lucyddm']['sigma'] == Sigma[0])]
yerr = std1sttruth / std1sttruth / np.sqrt(stdlist['N'])
ax.errorbar(stdlist['mu'], std1sttruth / std1sttruth, yerr=yerr, label=fr'$(20,0)$', marker='o', color='g')
for sigma, i in zip(Sigma, list(range(len(Sigma)))):
    for tau, j in zip(Tau, list(range(len(Tau)))):
        stdlist = mts['lucyddm'][(mts['lucyddm']['tau'] == tau) & (mts['lucyddm']['sigma'] == sigma)]
        sigma = int(sigma)
        tau = int(tau)
        yerr = stdlist['stdtruth'] / stdlist['std1sttruth'] / np.sqrt(stdlist['N'])
        ax.errorbar(stdlist['mu'], stdlist['stdtruth'] / stdlist['std1sttruth'], yerr=yerr, label=fr'$({tau},{sigma})$', marker=marker[i][j], color=colors[i][j])
ax.set_xlabel(r'$N_{\mathrm{PE}}\ \mathrm{expectation}\ \mu$')
ax.set_ylabel(r'$\mathrm{\sigma_\mathrm{ALL}/\sigma_\mathrm{1st}\ ratio}$')
ax.set_ylim(0.3, 1.1)
ax.grid()
ax.legend(title=r'$(\tau, \sigma)/\si{ns}$', bbox_to_anchor=(1., 0.9))
fig.savefig('Note/figures/vs-deltadiv.pgf')
fig.savefig('Note/figures/vs-deltadiv.pdf')
fig.savefig('Note/figures/vs-deltadiv.png')
plt.close(fig)

x = np.arange(0, len(keylist))
tau = 20
sigma = 5
wdist = np.vstack([mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]['wdist'].mean(axis=0) for key in keylist])
dy = np.vstack([wdist[:, 1] - wdist[:, 0], wdist[:, 2] - wdist[:, 1]])
fig = plt.figure(figsize=(10, 4))
fig.tight_layout()
ax = fig.add_subplot(111)
ax.bar(x, wdist[:, 1], color='b')
ax.set_ylim(0, math.ceil(wdist[~np.isnan(wdist)].max() + 0.5))
ax.set_ylabel(r'$\mathrm{Wasserstein\ Distance}/\si{ns}$')
ax.set_xticks(x)
ax.set_xticklabels(['$'+label[key]+'$' for key in keylist])
ax.errorbar(x, wdist[:, 1], yerr=dy, fmt='o', ecolor='r', color='b', elinewidth=2, capsize=3)
ax.set_title(fr'$\tau={tau}\si{{ns}},\,\sigma={sigma}\si{{ns}}$')
fig.savefig('Note/figures/summarycharge.pgf')
fig.savefig('Note/figures/summarycharge.pdf')
fig.savefig('Note/figures/summarycharge.png')
fig.clf()
plt.close(fig)

for key in mts.keys():
    print(key.rjust(10) + ' stdchargesuccess mean = {:.04%}'.format((mts[key]['stdchargesuccess'] / mts[key]['N']).mean()))
    print(key.rjust(10) + '  stdchargesuccess min = {:.04%}'.format((mts[key]['stdchargesuccess'] / mts[key]['N']).min()))
for m in ['mcmc', 'fbmp']:
    print((m + 'wave').rjust(10) + '   stdwavesuccess mean = {:.04%}'.format((mts[m]['stdwavesuccess'] / mts[m]['N']).mean()))
    print((m + 'wave').rjust(10) + '    stdwavesuccess min = {:.04%}'.format((mts[m]['stdwavesuccess'] / mts[m]['N']).min()))