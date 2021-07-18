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

filelist = os.listdir('result/fbmp/solu')
filelist = [f for f in filelist if f[0] != '.' and os.path.splitext(f)[-1] == '.h5']
numbers = [[float(i) for i in f[:-3].split('-')] for f in filelist]
stype = np.dtype([('mu', np.float64), ('tau', np.float64), ('sigma', np.float64), ('n', np.uint), ('std1sttruth', np.float64), ('stdtruth', np.float64), ('std', np.float64), ('stdone', np.float64), ('bias1sttruth', np.float64), ('biastruth', np.float64), ('bias', np.float64), ('biasone', np.float64), ('wdist', np.float64, 3), ('RSS', np.float64, 3), ('N', np.uint), ('stdsuccess', np.uint), ('stdonesuccess', np.uint)])
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
mtsi = np.sort(mtsi, kind='stable', order=['mu', 'tau', 'sigma'])

mts = {'findpeak':mtsi.copy(), 'threshold':mtsi.copy(), 'fftrans':mtsi.copy(), 'mcmc':mtsi.copy(), 'takara':mtsi.copy(), 'xiaopeip':mtsi.copy(), 'lucyddm':mtsi.copy(), 'fbmp':mtsi.copy()}
deltalabel = {'1st':'\mathrm{1st}', 'tru':'\mathrm{ALL}', 'findpeak':'\mathrm{FindPeak}', 'threshold':'\mathrm{Shift}', 'fftrans':'\mathrm{FFT}', 'lucyddm':'\mathrm{LucyDDM}', 'xiaopeip':'\mathrm{Fitting}', 'takara':'\mathrm{CNN}', 'fbmp':'\mathrm{FBMP}', 'fbmpone':'\mathrm{FBMPmax}', 'mcmc':'\mathrm{MCMCt0}', 'mcmcone':'\mathrm{MCMCcha}'}
label = {'1st':'\mathrm{1st}', 'tru':'\mathrm{ALL}', 'findpeak':'\mathrm{FindPeak}', 'threshold':'\mathrm{Shift}', 'fftrans':'\mathrm{FFT}', 'lucyddm':'\mathrm{LucyDDM}', 'xiaopeip':'\mathrm{Fitting}', 'takara':'\mathrm{CNN}', 'fbmp':'\mathrm{FBMP}', 'mcmc':'\mathrm{MCMC}'}
marker = {'1st':'o', 'tru':'h', 'findpeak':',', 'threshold':'1', 'fftrans':'+', 'lucyddm':'p', 'xiaopeip':'*', 'takara':'x', 'fbmp':'s', 'fbmpone':'^', 'mcmc':'X', 'mcmcone':'>'}
color = {'1st':'g', 'tru':'k', 'findpeak':'C1', 'threshold':'C2', 'fftrans':'m', 'lucyddm':'y', 'xiaopeip':'c', 'takara':'C0', 'fbmp':'r', 'fbmpone':'b', 'mcmc':'C4', 'mcmcone':'C5'}
jit = 0.05
jitter = {'mcmcone':-5 * jit, 'mcmc':-4 * jit, 'tru':-3 * jit, 'fbmp':-2 * jit, 'fbmpone':-1 * jit, 'lucyddm':0 * jit, 'xiaopeip':1 * jit, 'takara':2 * jit, '1st':3 * jit, 'fftrans':4 * jit, 'findpeak':5 * jit, 'threshold':6 * jit}

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
                r = wavef['SimTruth/T'].attrs['r']
            mts[key][i]['N'] = len(start)
            vali = np.abs(time['tscharge'] - start['T0'] - np.mean(time['tscharge'] - start['T0'])) <= r * np.std(time['tscharge'] - start['T0'], ddof=-1)
            mts[key][i]['std1sttruth'] = np.std(start['ts1sttruth'] - start['T0'], ddof=-1)
            mts[key][i]['stdtruth'] = np.std(start['tstruth'] - start['T0'], ddof=-1)
            mts[key][i]['std'], mts[key][i]['stdsuccess'] = wff.stdrmoutlier(time['tscharge'] - start['T0'], r)
            mts[key][i]['bias1sttruth'] = np.mean(start['ts1sttruth'] - start['T0'])
            mts[key][i]['biastruth'] = np.mean(start['tstruth'] - start['T0'])
            mts[key][i]['bias'] = np.mean(time['tscharge'][vali] - start['T0'][vali])
            mts[key][i]['wdist'] = np.insert(np.percentile(record['wdist'][vali], [5, 95]), 1, record['wdist'][vali].mean())
            mts[key][i]['RSS'] = np.insert(np.percentile(record['RSS'][vali], [5, 95]), 1, record['RSS'][vali].mean())
            if not np.any(np.isnan(time['tswave'][vali])):
                mts[key][i]['stdone'], mts[key][i]['stdonesuccess'] = wff.stdrmoutlier(time['tscharge'] - start['T0'], r)
                mts[key][i]['biasone'] = np.mean(time['tscharge'][vali] - start['T0'][vali])
                vali = np.abs(time['tswave'] - start['T0'] - np.mean(time['tswave'] - start['T0'])) < r * np.std(time['tswave'] - start['T0'], ddof=-1)
                mts[key][i]['std'], mts[key][i]['stdsuccess'] = wff.stdrmoutlier(time['tswave'] - start['T0'], r)
                mts[key][i]['bias'] = np.mean(time['tswave'][vali] - start['T0'][vali])
        except:
            pass

dhigh = np.array([[np.max(mts[key]['std1sttruth']), np.max(mts[key]['stdtruth']), np.max(mts[key]['std']), np.max(mts[key]['stdone'])] for key in mts.keys()])
dhigh = np.max(dhigh[~np.isnan(dhigh)]) * 1.05
whigh = np.array([[np.max(mts[key]['wdist'])] for key in mts.keys()])
whigh = np.max(whigh[~np.isnan(whigh)]) * 1.05
rhigh = np.array([[np.max(mts[key]['RSS'])] for key in mts.keys()])
rhigh = np.max(rhigh[~np.isnan(rhigh)]) * 1.05

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
alpha = 0.05
lim = {'deltadiv':np.array([[0.3, 0.5]]), 'wdist':np.array([[1.5, 3.0]]), 'rss':np.array([[0.55e3, 2.2e3]])}
keylist = list(mts.keys())
badkey = ['findpeak', 'threshold', 'fftrans', 'mcmc']
for sigma, i in zip(Sigma, list(range(len(Sigma)))):
    for tau, j in zip(Tau, list(range(len(Tau)))):
        ax = figd.add_subplot(gsd[i, j])
        stdlist = mts['lucyddm'][(mts['lucyddm']['tau'] == tau) & (mts['lucyddm']['sigma'] == sigma)]
        yerr1st = np.vstack([stdlist['std1sttruth']-np.sqrt(np.power(stdlist['std1sttruth'],2)*stdlist['N']/chi2.ppf(1-alpha/2, stdlist['N'])), np.sqrt(np.power(stdlist['std1sttruth'],2)*stdlist['N']/chi2.ppf(alpha/2, stdlist['N']))-stdlist['std1sttruth']])
        yerrall = np.vstack([stdlist['stdtruth']-np.sqrt(np.power(stdlist['stdtruth'],2)*stdlist['N']/chi2.ppf(1-alpha/2, stdlist['N'])), np.sqrt(np.power(stdlist['stdtruth'],2)*stdlist['N']/chi2.ppf(alpha/2, stdlist['N']))-stdlist['stdtruth']])
        ax.errorbar(stdlist['mu'] + jitter['1st'], stdlist['std1sttruth'], yerr=yerr1st, c=color['1st'], label='$\sigma_'+deltalabel['1st']+'$', marker=marker['1st'])
        ax.errorbar(stdlist['mu'] + jitter['tru'], stdlist['stdtruth'], yerr=yerrall, c=color['tru'], label='$\sigma_'+deltalabel['tru']+'$', marker=marker['tru'])
        for m in ['mcmc', 'fbmp']:
            stdlistwav = mts[m][(mts[m]['tau'] == tau) & (mts[m]['sigma'] == sigma)]
            yerrwav = np.vstack([stdlistwav['stdone']-np.sqrt(np.power(stdlistwav['stdone'],2)*stdlistwav['N']/chi2.ppf(1-alpha/2, stdlistwav['N'])), np.sqrt(np.power(stdlistwav['stdone'],2)*stdlistwav['N']/chi2.ppf(alpha/2, stdlistwav['N']))-stdlistwav['stdone']])
            ax.errorbar(stdlistwav['mu'] + jitter[m + 'one'], stdlistwav['stdone'], yerr=yerrwav, c=color[m + 'one'], label='$\sigma_'+deltalabel[m + 'one']+'$', marker=marker[m + 'one'])
        for key in keylist:
            stdlist = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
            yerrcha = np.vstack([stdlist['std']-np.sqrt(np.power(stdlist['std'],2)*stdlist['N']/chi2.ppf(1-alpha/2, stdlist['N'])), np.sqrt(np.power(stdlist['std'],2)*stdlist['N']/chi2.ppf(alpha/2, stdlist['N']))-stdlist['std']])
            ax.errorbar(stdlist['mu'] + jitter[key], stdlist['std'], yerr=yerrcha, c=color[key], label='$\sigma_'+deltalabel[key]+'$', marker=marker[key])
        ax.set_xlabel(r'$N_{\mathrm{PE}}\ \mathrm{expectation}\ \mu$')
        ax.set_ylabel(r'$\sigma/\si{ns}$')
        ax.set_title(fr'$\tau={tau}\si{{ns}},\,\sigma={sigma}\si{{ns}}$')
        # ax.set_yscale('log')
        ax.grid()
        if i == len(Sigma) - 1 and j == len(Tau) - 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1., 0.9))

        ax = figb.add_subplot(gsb[i, j])
        stdlist = mts['lucyddm'][(mts['lucyddm']['tau'] == tau) & (mts['lucyddm']['sigma'] == sigma)]
        # yerr1st = np.vstack([stdlist['bias1sttruth']-t.ppf(1-alpha/2, stdlist['N'])*stdlist['std1sttruth']/np.sqrt(stdlist['N']), t.ppf(1-alpha/2, stdlist['N'])*stdlist['std1sttruth']/np.sqrt(stdlist['N'])-stdlist['bias1sttruth']])
        yerrall = np.vstack([stdlist['biastruth']-t.ppf(1-alpha/2, stdlist['N'])*stdlist['stdtruth']/np.sqrt(stdlist['N']), t.ppf(1-alpha/2, stdlist['N'])*stdlist['stdtruth']/np.sqrt(stdlist['N'])-stdlist['biastruth']])
        # ax.errorbar(stdlist['mu'] + jitter['1st'], stdlist['bias1sttruth'], yerr=yerr1st, c=color['1st'], label='$\mathrm{bias}_'+deltalabel['1st']+'$', marker=marker['1st'])
        ax.errorbar(stdlist['mu'] + jitter['tru'], stdlist['biastruth'], yerr=yerrall, c=color['tru'], label='$\mathrm{bias}_'+deltalabel['tru']+'$', marker=marker['tru'])
        for key in keylist:
            if key in badkey:
                continue
            stdlist = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
            yerrcha = np.vstack([stdlist['bias']-t.ppf(1-alpha/2, stdlist['N'])*stdlist['std']/np.sqrt(stdlist['N']), t.ppf(1-alpha/2, stdlist['N'])*stdlist['std']/np.sqrt(stdlist['N'])-stdlist['bias']])
            ax.errorbar(stdlist['mu'] + jitter[key], stdlist['bias'], yerr=yerrcha, c=color[key], label='$\mathrm{bias}_'+deltalabel[key]+'$', marker=marker[key])
        ax.set_xlabel(r'$N_{\mathrm{PE}}\ \mathrm{expectation}\ \mu$')
        ax.set_ylabel(r'$\mathrm{bias}/\si{ns}$')
        ax.set_title(fr'$\tau={tau}\si{{ns}},\,\sigma={sigma}\si{{ns}}$')
        # ax.set_yscale('log')
        ax.grid()
        if i == len(Sigma) - 1 and j == len(Tau) - 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1., 0.9))

        ax = figdd.add_subplot(gsdd[i, j])
        stdlist = mts['lucyddm'][(mts['lucyddm']['tau'] == tau) & (mts['lucyddm']['sigma'] == sigma)]
        for key in keylist:
            if key in badkey:
                continue
            stdlistkey = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
            yerr = stdlistkey['std'] / stdlist['stdtruth'] / np.sqrt(stdlistkey['N'])
            ax.errorbar(stdlist['mu'] + jitter[key], stdlistkey['std'] / stdlist['stdtruth'], yerr=yerr, label='$\sigma_'+deltalabel[key]+'/\sigma_'+deltalabel['tru']+'$', c=color[key], marker=marker[key])
        ax.set_xlabel(r'$N_{\mathrm{PE}}\ \mathrm{expectation}\ \mu$')
        ax.set_ylabel(r'$\mathrm{ratio}$')
        ax.set_title(fr'$\tau={tau}\si{{ns}},\,\sigma={sigma}\si{{ns}}$')
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
        ax.set_xlabel(r'$N_{\mathrm{PE}}\ \mathrm{expectation}\ \mu$')
        ax.set_ylabel(r'$\mathrm{W-dist}/\si{ns}$')
        ax.set_title(fr'$\tau={tau}\si{{ns}},\,\sigma={sigma}\si{{ns}}$')
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
        ax.set_xlabel(r'$N_{\mathrm{PE}}\ \mathrm{expectation}\ \mu$')
        ax.set_ylabel(r'$\mathrm{RSS}/\si{mV}^{2}$')
        ax.set_title(fr'$\tau={tau}\si{{ns}},\,\sigma={sigma}\si{{ns}}$')
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

alpha = 0.05
marker = [['s', '^']]
colors = [['r', 'b']]
fig = plt.figure(figsize=(8, 4))
gs = gridspec.GridSpec(1, 2, figure=fig, left=0.1, right=0.85, top=0.92, bottom=0.15, wspace=0.25, hspace=0.2)
ax = fig.add_subplot(gs[0, 0])
std1sttruth = np.empty(len(stdlist['mu']))
stdlist = mts['lucyddm'][(mts['lucyddm']['tau'] == Tau[0]) & (mts['lucyddm']['sigma'] == Sigma[0])]
sigma = 0
tau = max(Tau)
np.random.seed(0)
for mu, i in zip(stdlist['mu'], list(range(len(stdlist['mu'])))):
    N = stdlist['N'][i]
    npe = poisson.ppf(1 - uniform.rvs(scale=1-poisson.cdf(0, mu), size=N), mu).astype(int)
    t0 = np.random.uniform(100., 500., size=N)
    sams = [wff.time(npe[j], tau, sigma) + t0[j] for j in range(N)]
    ts1sttruth = np.array([np.min(sams[j]) for j in range(N)])
    std1sttruth[i] = np.std(ts1sttruth - t0, ddof=-1)
yerr = np.vstack([std1sttruth-np.sqrt(np.power(std1sttruth,2)*stdlist['N']/chi2.ppf(1-alpha/2, stdlist['N'])), np.sqrt(np.power(std1sttruth,2)*stdlist['N']/chi2.ppf(alpha/2, stdlist['N']))-std1sttruth])
ax.errorbar(stdlist['mu'], std1sttruth, yerr=yerr, label='$\sigma_'+deltalabel['tru']+fr',\,\tau={tau}\si{{ns}},\,\sigma={sigma}\si{{ns}}$', marker='o', color='g')
ax.errorbar(stdlist['mu'], std1sttruth, yerr=yerr, label='$\sigma_'+deltalabel['1st']+fr',\,\tau={tau}\si{{ns}},\,\sigma={sigma}\si{{ns}}$', marker='o', color='g', linestyle='dashed')
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
tau = max(Tau)
stdlist = mts['lucyddm'][(mts['lucyddm']['tau'] == Tau[0]) & (mts['lucyddm']['sigma'] == Sigma[0])]
yerr = std1sttruth / std1sttruth / np.sqrt(stdlist['N'])
ax.errorbar(stdlist['mu'], std1sttruth / std1sttruth, yerr=yerr, label=fr'$(20,0)$', marker='o', color='g')
for sigma, i in zip(Sigma, list(range(len(Sigma)))):
    for tau, j in zip(Tau, list(range(len(Tau)))):
        stdlist = mts['lucyddm'][(mts['lucyddm']['tau'] == tau) & (mts['lucyddm']['sigma'] == sigma)]
        sigma = int(sigma)
        tau = int(tau)
        yerr = stdlist['std1sttruth'] / stdlist['stdtruth'] / np.sqrt(stdlist['N'])
        ax.errorbar(stdlist['mu'], stdlist['std1sttruth'] / stdlist['stdtruth'], yerr=yerr, label=fr'$({tau},{sigma})$', marker=marker[i][j], color=colors[i][j])
ax.set_xlabel(r'$N_{\mathrm{PE}}\ \mathrm{expectation}\ \mu$')
ax.set_ylabel(r'$\mathrm{\sigma_\mathrm{1st}/\sigma_\mathrm{ALL}\ ratio}$')
ax.set_ylim(0.9, 3.0)
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
    print(key.rjust(10) + ' stdchargesuccess mean = {:.04%}'.format((mts[key]['stdsuccess'] / mts[key]['N']).mean()))
    print(key.rjust(10) + '  stdchargesuccess min = {:.04%}'.format((mts[key]['stdsuccess'] / mts[key]['N']).min()))
for m in ['mcmc', 'fbmp']:
    print((m + 'one').rjust(10) + '   stdwavesuccess mean = {:.04%}'.format((mts[m]['stdonesuccess'] / mts[m]['N']).mean()))
    print((m + 'one').rjust(10) + '    stdwavesuccess min = {:.04%}'.format((mts[m]['stdonesuccess'] / mts[m]['N']).min()))

stype = np.dtype([('mu', np.float64), ('tau', np.float64), ('sigma', np.float64), ('n', np.uint), ('stdmutru', np.float64), ('stdmuint', np.float64), ('stdmupe', np.float64), ('stdmumax', np.float64), ('stdmu', np.float64), ('biasmuint', np.float64), ('biasmupe', np.float64), ('biasmumax', np.float64), ('biasmu', np.float64), ('N', np.uint)])
mtsi = np.zeros(len(numbers), dtype=stype)
mtsi['mu'] = np.array([i[0] for i in numbers])
mtsi['tau'] = np.array([i[1] for i in numbers])
mtsi['sigma'] = np.array([i[2] for i in numbers])
mtsi['n'] = np.arange(len(numbers))
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

mts = {'fbmp':mtsi.copy()}

marker = {'int':'o', 'tru':'h', 'pe':'p', 'fbmp':'s', 'max':'^'}
color = {'int':'g', 'tru':'k', 'pe':'y', 'fbmp':'r', 'max':'b'}

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
                pelist = wavef['SimTriggerInfo/PEList'][:]
                waves = wavef['Readout/Waveform'][:]
                gmu = wavef['SimTriggerInfo/PEList'].attrs['gmu']
                gsigma = wavef['SimTriggerInfo/PEList'].attrs['gsigma']
                r = wavef['SimTruth/T'].attrs['r']
            mts[key][i]['N'] = len(start)
            vali = np.abs(time['tscharge'] - start['T0'] - np.mean(time['tscharge'] - start['T0'])) <= r * np.std(time['tscharge'] - start['T0'], ddof=-1)
            mts[key][i]['N'] = len(start)
            Chnum = len(np.unique(pelist['PMTId']))
            e_ans, i_ans = np.unique(pelist['TriggerNo'] * Chnum + pelist['PMTId'], return_index=True)
            i_ans = np.append(i_ans, len(pelist))
            pe_sum = np.array([pelist[i_ans[i]:i_ans[i+1]]['Charge'].sum() for i in range(len(e_ans))]) / gmu
            wave_sum = waves['Waveform'].sum(axis=1) / gmu
            n = np.arange(1, 1000)
            mean = np.average(n, weights=poisson.pmf(n, mu=mu))
            lognm = np.average(np.log(n), weights=poisson.pmf(n, mu=mu))
            slog = np.sqrt(np.average((np.log(n) - lognm)**2, weights=poisson.pmf(n, mu=mu)))
            mts[key][i]['stdmutru'] = slog
            slog = np.std(np.log(wave_sum[vali]), ddof=-1)
            m = np.mean(wave_sum[vali]) - mean
            mts[key][i]['stdmuint'] = slog
            mts[key][i]['biasmuint'] = m
            s = np.std(pe_sum[vali], ddof=-1)
            slog = np.std(np.log(pe_sum[vali]), ddof=-1)
            m = np.mean(pe_sum[vali]) - mean
            mts[key][i]['stdmupe'] = slog
            mts[key][i]['biasmupe'] = m
            s = np.std(time['mucharge'][vali], ddof=-1)
            slog = np.std(np.log(time['mucharge'][vali]), ddof=-1)
            m = np.mean(time['mucharge'][vali]) - mean
            mts[key][i]['stdmumax'] = slog
            mts[key][i]['biasmumax'] = m
            s = np.std(time['muwave'][vali], ddof=-1)
            slog = np.std(np.log(time['muwave'][vali]), ddof=-1)
            m = np.mean(time['muwave'][vali]) - mean
            mts[key][i]['stdmu'] = slog
            mts[key][i]['biasmu'] = m
        except:
            pass

key = 'fbmp'
figdd = plt.figure(figsize=(len(Tau) * 5, len(Sigma) * 3))
gs = gridspec.GridSpec(1, 2, figure=figdd, left=0.1, right=0.8, top=0.92, bottom=0.15, wspace=0.3, hspace=0.35)
figbr = plt.figure(figsize=(len(Tau) * 5, len(Sigma) * 3))
gs = gridspec.GridSpec(1, 2, figure=figbr, left=0.1, right=0.8, top=0.92, bottom=0.15, wspace=0.3, hspace=0.35)
figb = plt.figure(figsize=(len(Tau) * 5, len(Sigma) * 3))
gs = gridspec.GridSpec(1, 2, figure=figb, left=0.1, right=0.8, top=0.92, bottom=0.15, wspace=0.3, hspace=0.35)
for sigma, i in zip(Sigma, list(range(len(Sigma)))):
    for tau, j in zip(Tau, list(range(len(Tau)))):
        stdlistkey = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
        ax = figdd.add_subplot(gs[i, j])
        yerr = stdlistkey['stdmuint'] / stdlistkey['stdmutru'] / np.sqrt(2 * stdlistkey['N'])
        ax.errorbar(stdlistkey['mu'], stdlistkey['stdmuint'] / stdlistkey['stdmutru'], yerr=yerr, label='$\sigma_\mathrm{int}/\sigma_{\log{\mu}}$', c=color['int'], marker=marker['int'])
        # yerr = stdlistkey['stdmupe'] / stdlistkey['stdmutru'] / np.sqrt(2 * stdlistkey['N'])
        # ax.errorbar(stdlistkey['mu'], stdlistkey['stdmupe'] / stdlistkey['stdmutru'], yerr=yerr, label='$\sigma_\mathrm{pe}/\sigma_{\log{\mu}}$', c=color['pe'], marker=marker['pe'])
        yerr = stdlistkey['stdmumax'] / stdlistkey['stdmutru'] / np.sqrt(2 * stdlistkey['N'])
        ax.errorbar(stdlistkey['mu'], stdlistkey['stdmumax'] / stdlistkey['stdmutru'], yerr=yerr, label='$\sigma_\mathrm{FBMPmax}/\sigma_{\log{\mu}}$', c=color['max'], marker=marker['max'])
        yerr = stdlistkey['stdmu'] / stdlistkey['stdmutru'] / np.sqrt(2 * stdlistkey['N'])
        ax.errorbar(stdlistkey['mu'], stdlistkey['stdmu'] / stdlistkey['stdmutru'], yerr=yerr, label='$\sigma_\mathrm{FBMP}/\sigma_{\log{\mu}}$', c=color['fbmp'], marker=marker['fbmp'])
        ax.set_xlabel(r'$N_{\mathrm{PE}}\ \mathrm{expectation}\ \mu$')
        ax.set_ylabel(r'$\mathrm{ratio}$')
        ax.set_title(fr'$\tau={tau}\si{{ns}},\,\sigma={sigma}\si{{ns}}$')
        ax.set_ylim(0.97, 1.52)
        ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
        ax.grid()
        if i == len(Sigma) - 1 and j == len(Tau) - 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1., 0.9))
        
        ax = figbr.add_subplot(gs[i, j])
        n = np.arange(1, 1000)
        mean = np.array([np.average(n, weights=poisson.pmf(n, mu=mu)) for mu in stdlistkey['mu']])
        yerr = stdlistkey['biasmuint'] / np.sqrt(stdlistkey['N']) / mean
        ax.errorbar(stdlistkey['mu'], stdlistkey['biasmuint'] / mean, yerr=yerr, label='$\mathrm{bias}_\mathrm{int}$', c=color['int'], marker=marker['int'])
        # yerr = stdlistkey['biasmupe'] / np.sqrt(stdlistkey['N']) / mean
        # ax.errorbar(stdlistkey['mu'], stdlistkey['biasmupe'] / mean, yerr=yerr, label='$\mathrm{bias}_\mathrm{pe}$', c=color['pe'], marker=marker['pe'])
        yerr = stdlistkey['biasmumax'] / np.sqrt(stdlistkey['N']) / mean
        ax.errorbar(stdlistkey['mu'], stdlistkey['biasmumax'] / mean, yerr=yerr, label='$\mathrm{bias}_\mathrm{FBMPmax}$', c=color['max'], marker=marker['max'])
        yerr = stdlistkey['biasmu'] / np.sqrt(stdlistkey['N']) / mean
        ax.errorbar(stdlistkey['mu'], stdlistkey['biasmu'] / mean, yerr=yerr, label='$\mathrm{bias}_\mathrm{FBMP}$', c=color['fbmp'], marker=marker['fbmp'])
        ax.set_xlabel(r'$N_{\mathrm{PE}}\ \mathrm{expectation}\ \mu$')
        # ax.set_ylabel(r'$\mathrm{bias}$')
        ax.set_ylabel(r'$\frac{\Delta \mu}{\mu}$')
        ax.set_title(fr'$\tau={tau}\si{{ns}},\,\sigma={sigma}\si{{ns}}$')
        # ax.set_ylim(-0.03, 0.025)
        ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
        ax.grid()
        if i == len(Sigma) - 1 and j == len(Tau) - 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1., 0.9))

        ax = figb.add_subplot(gs[i, j])
        n = np.arange(1, 1000)
        yerr = stdlistkey['biasmuint'] / np.sqrt(stdlistkey['N'])
        ax.errorbar(stdlistkey['mu'], stdlistkey['biasmuint'], yerr=yerr, label='$\mathrm{bias}_\mathrm{int}$', c=color['int'], marker=marker['int'])
        # yerr = stdlistkey['biasmupe'] / np.sqrt(stdlistkey['N'])
        # ax.errorbar(stdlistkey['mu'], stdlistkey['biasmupe'], yerr=yerr, label='$\mathrm{bias}_\mathrm{pe}$', c=color['pe'], marker=marker['pe'])
        yerr = stdlistkey['biasmumax'] / np.sqrt(stdlistkey['N'])
        ax.errorbar(stdlistkey['mu'], stdlistkey['biasmumax'], yerr=yerr, label='$\mathrm{bias}_\mathrm{FBMPmax}$', c=color['max'], marker=marker['max'])
        yerr = stdlistkey['biasmu'] / np.sqrt(stdlistkey['N'])
        ax.errorbar(stdlistkey['mu'], stdlistkey['biasmu'], yerr=yerr, label='$\mathrm{bias}_\mathrm{FBMP}$', c=color['fbmp'], marker=marker['fbmp'])
        ax.set_xlabel(r'$N_{\mathrm{PE}}\ \mathrm{expectation}\ \mu$')
        # ax.set_ylabel(r'$\mathrm{bias}$')
        ax.set_ylabel(r'$\Delta \mu$')
        ax.set_title(fr'$\tau={tau}\si{{ns}},\,\sigma={sigma}\si{{ns}}$')
        ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
        ax.grid()
        if i == len(Sigma) - 1 and j == len(Tau) - 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1., 0.9))
figdd.savefig('Note/figures/vs-deltamethodsdivmu.pgf')
figdd.savefig('Note/figures/vs-deltamethodsdivmu.pdf')
figdd.savefig('Note/figures/vs-deltamethodsdivmu.png')
plt.close(figdd)
figbr.savefig('Note/figures/vs-biasmu.pgf')
figbr.savefig('Note/figures/vs-biasmu.pdf')
figbr.savefig('Note/figures/vs-biasmu.png')
plt.close(figbr)
figb.savefig('Note/figures/vs-biasmut.pgf')
figb.savefig('Note/figures/vs-biasmut.pdf')
figb.savefig('Note/figures/vs-biasmut.png')
plt.close(figb)