import os
import sys
import re
import time
import math
import csv
import argparse
import warnings
warnings.filterwarnings('ignore')

import h5py
import numpy as np
from scipy import stats
from scipy.stats import chi2
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec

import wf_func as wff

import matplotlib
matplotlib.use('pgf')

plt.style.use('default')
plt.rcParams['font.size'] = 15

psr = argparse.ArgumentParser()
psr.add_argument('--conf', type=str, help='configuration of tau & sigma')
args = psr.parse_args()

with open(args.conf) as f:
    f_csv = csv.reader(f, delimiter=' ')
    Tau = next(f_csv)
    Tau = [int(i) for i in Tau][:2]
    Sigma = next(f_csv)
    Sigma = [int(i) for i in Sigma]

filelist = os.listdir('result/lucyddm/solu')
filelist = [f for f in filelist if f[0] != '.' and os.path.splitext(f)[-1] == '.h5']
numbers = [[float(i) for i in f[:-3].split('-')] for f in filelist]
stype = np.dtype([('mu', np.float64), ('tau', np.float64), ('sigma', np.float64), ('n', np.uint), ('std1sttruth', np.float64), ('stdtruth', np.float64), ('stdcharge', np.float64), ('stdwave', np.float64), ('wdist', np.float64, 3), ('RSS', np.float64, 3), ('N', np.uint), ('stdchargesuccess', np.uint), ('stdwavesuccess', np.uint)])
mtsi = np.zeros(len(numbers), dtype=stype)
mtsi['mu'] = np.array([i[0] for i in numbers])
mtsi['tau'] = np.array([i[1] for i in numbers])
mtsi['sigma'] = np.array([i[2] for i in numbers])
mtsi['n'] = np.arange(len(numbers))
mtsi['std1sttruth'] = np.nan
mtsi['stdtruth'] = np.nan
mtsi['stdcharge'] = np.nan
mtsi['stdwave'] = np.nan
mtsi['wdist'] = np.nan
mtsi['RSS'] = np.nan
mtsi = np.sort(mtsi, kind='stable', order=['mu', 'tau', 'sigma'])

mts = {'findpeak':mtsi.copy(), 'threshold':mtsi.copy(), 'fftrans':mtsi.copy(), 'lucyddm':mtsi.copy(), 'xiaopeip':mtsi.copy(), 'mcmc':mtsi.copy(), 'takara':mtsi.copy(), 'fbmp':mtsi.copy()}
deltalabel = {'1st':'\mathrm{1st}', 'tru':'\mathrm{Truth}', 'findpeak':'\mathrm{FindPeak}', 'threshold':'\mathrm{Shift}', 'fftrans':'\mathrm{FFT}', 'lucyddm':'\mathrm{LucyDDM}', 'xiaopeip':'\mathrm{Fitting}', 'takara':'\mathrm{CNN}', 'fbmp':'\mathrm{FBMPcha}', 'fbmpwave':'\mathrm{FBMPt0}', 'mcmc':'\mathrm{MCMCcha}', 'mcmcwave':'\mathrm{MCMCt0}'}
label = {'1st':'\mathrm{1st}', 'tru':'\mathrm{Truth}', 'findpeak':'\mathrm{FindPeak}', 'threshold':'\mathrm{Shift}', 'fftrans':'\mathrm{FFT}', 'lucyddm':'\mathrm{LucyDDM}', 'xiaopeip':'\mathrm{Fitting}', 'takara':'\mathrm{CNN}', 'fbmp':'\mathrm{FBMP}', 'fbmpwave':'\mathrm{FBMP}', 'mcmc':'\mathrm{MCMC}', 'mcmcwave':'\mathrm{MCMC}'}
color = {'1st':'b', 'tru':'k', 'findpeak':'C1', 'threshold':'C2', 'fftrans':'m', 'lucyddm':'y', 'xiaopeip':'c', 'takara':'C0', 'fbmp':'r', 'fbmpwave':'b', 'mcmc':'C4', 'mcmcwave':'C5'}
jit = 0.05
jitter = {'tru':-6 * jit, 'mcmc':-5 * jit, 'fbmp':-4 * jit, 'fbmpwave':-3 * jit, 'lucyddm':-2 * jit, 'mcmcwave':-1 * jit, 'xiaopeip':1 * jit, 'takara':0 * jit, 'fftrans':2 * jit, '1st':3 * jit, 'findpeak':4 * jit, 'threshold':5 * jit}

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
            vali = np.abs(time['tscharge'] - start['T0'] - np.mean(time['tscharge'] - start['T0'])) < r * np.std(time['tscharge'] - start['T0'], ddof=-1)
            mts[key][i]['N'] = len(start)
            mts[key][i]['std1sttruth'] = np.std(start['ts1sttruth'] - start['T0'], ddof=-1)
            mts[key][i]['stdtruth'] = np.std(start['tstruth'] - start['T0'], ddof=-1)
            mts[key][i]['stdcharge'], mts[key][i]['stdchargesuccess'] = wff.stdrmoutlier(time['tscharge'][vali] - start['T0'][vali], r)
            if not np.any(np.isnan(time['tswave'][vali])):
                mts[key][i]['stdwave'], mts[key][i]['stdwavesuccess'] = wff.stdrmoutlier(time['tswave'][vali] - start['T0'][vali], r)
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

figd = plt.figure(figsize=(len(Sigma) * 6, len(Tau) * 4))
gsd = gridspec.GridSpec(len(Tau), len(Sigma), figure=figd, left=0.1, right=0.8, top=0.95, bottom=0.1, wspace=0.2, hspace=0.3)
figdd = plt.figure(figsize=(len(Sigma) * 6, len(Tau) * 4))
gsdd = gridspec.GridSpec(len(Tau), len(Sigma), figure=figd, left=0.1, right=0.8, top=0.95, bottom=0.1, wspace=0.2, hspace=0.3)
figw = plt.figure(figsize=(len(Sigma) * 6, len(Tau) * 4))
gsw = gridspec.GridSpec(len(Tau), len(Sigma), figure=figw, left=0.1, right=0.8, top=0.95, bottom=0.1, wspace=0.2, hspace=0.3)
figr = plt.figure(figsize=(len(Sigma) * 6, len(Tau) * 4))
gsr = gridspec.GridSpec(len(Tau), len(Sigma), figure=figr, left=0.1, right=0.8, top=0.95, bottom=0.1, wspace=0.2, hspace=0.3)
alpha = 0.05
lim = {'deltadiv':np.tile([0.3, 0.5, 0.], (2, 1)), 'wdist':np.tile([2, 3.5, 7], (2, 1)), 'rss':np.array([[5.5e3, 3e3, 2e3], [4.5e3, 3e3, 2e3]])}
keylist = list(mts.keys())
for sigma, i in zip(Sigma, list(range(len(Sigma)))):
    for tau, j in zip(Tau, list(range(len(Tau)))):
        ax = figd.add_subplot(gsd[j, i])
        stdlist = mts['lucyddm'][(mts['lucyddm']['tau'] == tau) & (mts['lucyddm']['sigma'] == sigma)]
        yerr1st = np.vstack([stdlist['std1sttruth']-np.sqrt(np.power(stdlist['std1sttruth'],2)*stdlist['N']/chi2.ppf(1-alpha/2, stdlist['N'])), np.sqrt(np.power(stdlist['std1sttruth'],2)*stdlist['N']/chi2.ppf(alpha/2, stdlist['N']))-stdlist['std1sttruth']])
        yerrall = np.vstack([stdlist['stdtruth']-np.sqrt(np.power(stdlist['stdtruth'],2)*stdlist['N']/chi2.ppf(1-alpha/2, stdlist['N'])), np.sqrt(np.power(stdlist['stdtruth'],2)*stdlist['N']/chi2.ppf(alpha/2, stdlist['N']))-stdlist['stdtruth']])
        ax.errorbar(stdlist['mu'] + jitter['1st'], stdlist['std1sttruth'], yerr=yerr1st, c=color['1st'], label='$\delta_'+deltalabel['1st']+'$', marker='^')
        ax.errorbar(stdlist['mu'] + jitter['tru'], stdlist['stdtruth'], yerr=yerrall, c=color['tru'], label='$\delta_'+deltalabel['tru']+'$', marker='^')
        for m in ['mcmc', 'fbmp']:
            stdlistwav = mts[m][(mts[m]['tau'] == tau) & (mts[m]['sigma'] == sigma)]
            yerrwav = np.vstack([stdlistwav['stdwave']-np.sqrt(np.power(stdlistwav['stdwave'],2)*stdlistwav['N']/chi2.ppf(1-alpha/2, stdlistwav['N'])), np.sqrt(np.power(stdlistwav['stdwave'],2)*stdlistwav['N']/chi2.ppf(alpha/2, stdlistwav['N']))-stdlistwav['stdwave']])
            ax.errorbar(stdlistwav['mu'] + jitter[m + 'wave'], stdlistwav['stdwave'], yerr=yerrwav, c=color[m + 'wave'], label='$\delta_'+deltalabel[m + 'wave']+'$', marker='^')
        for k in range(len(keylist)):
            key = keylist[k]
            stdlist = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
            yerrcha = np.vstack([stdlist['stdcharge']-np.sqrt(np.power(stdlist['stdcharge'],2)*stdlist['N']/chi2.ppf(1-alpha/2, stdlist['N'])), np.sqrt(np.power(stdlist['stdcharge'],2)*stdlist['N']/chi2.ppf(alpha/2, stdlist['N']))-stdlist['stdcharge']])
            ax.errorbar(stdlist['mu'] + jitter[key], stdlist['stdcharge'], yerr=yerrcha, c=color[key], label='$\delta_'+deltalabel[key]+'$', marker='^')
        ax.set_xlabel(r'$\mu$')
        ax.set_ylabel(r'$\delta/\mathrm{ns}$')
        ax.set_title(fr'$\tau={tau}\mathrm{{ns}},\,\sigma={sigma}\mathrm{{ns}}$')
        # ax.set_yscale('log')
        ax.grid()
        if i == len(Sigma) - 1 and j == len(Tau) - 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1., 1.5))

        ax = figdd.add_subplot(gsdd[j, i])
        stdlist = mts['lucyddm'][(mts['lucyddm']['tau'] == tau) & (mts['lucyddm']['sigma'] == sigma)]
        yerrall = stdlist['stdtruth'] / stdlist['std1sttruth'] / np.sqrt(stdlist['N'])
        ax.errorbar(stdlist['mu'] + jitter['tru'], stdlist['stdtruth'] / stdlist['std1sttruth'], yerr=yerrall, label='$\delta_'+deltalabel['tru']+'/\delta_'+deltalabel['1st']+'$', c=color['tru'], marker='^')
        # ax.plot(stdlist['mu'] + jitter['tru'], stdlist['stdtruth'] / stdlist['std1sttruth'], label='$\delta_'+deltalabel['tru']+'/\delta_'+deltalabel['1st']+'$', c=color['tru'], marker='^')
        for k in range(len(keylist)):
            key = keylist[k]
            if key == 'findpeak' or key == 'threshold' or key == 'fftrans':
                continue
            stdlistkey = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
            yerr = stdlistkey['stdcharge'] / stdlist['std1sttruth'] / np.sqrt(stdlistkey['N'])
            ax.errorbar(stdlist['mu'] + jitter[key], stdlistkey['stdcharge'] / stdlist['std1sttruth'], yerr=yerr, label='$\delta_'+deltalabel[key]+'/\delta_'+deltalabel['1st']+'$', c=color[key], marker='^')
            # ax.plot(stdlist['mu'] + jitter[key], stdlistkey['stdcharge'] / stdlist['std1sttruth'], label='$\delta_'+deltalabel[key]+'/\delta_'+deltalabel['1st']+'$', c=color[key], marker='^')
        for m in ['mcmc', 'fbmp']:
            stdlistwav = mts[m][(mts[m]['tau'] == tau) & (mts[m]['sigma'] == sigma)]
            yerrwav = stdlistwav['stdcharge'] / stdlist['std1sttruth'] / np.sqrt(stdlistwav['N'])
            ax.errorbar(stdlistwav['mu'] + jitter[m + 'wave'], stdlistwav['stdwave'] / stdlist['std1sttruth'], yerr=yerrwav, label='$\delta_'+deltalabel[m + 'wave']+'/\delta_'+deltalabel['1st']+'$', c=color[m + 'wave'], marker='^')
            # ax.plot(stdlistwav['mu'] + jitter[m + 'wave'], stdlistwav['stdwave'] / stdlist['std1sttruth'], label='$\delta_'+deltalabel[m + 'wave']+'/\delta_'+deltalabel['1st']+'$', c=color[m + 'wave'], marker='^')
        ax.set_xlabel(r'$\mu$')
        ax.set_ylabel(r'$ratio$')
        ax.set_title(fr'$\tau={tau}\mathrm{{ns}},\,\sigma={sigma}\mathrm{{ns}}$')
        ax.set_ylim(lim['deltadiv'][i, j], 1.01)
        ax.grid()
        if i == len(Sigma) - 1 and j == len(Tau) - 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1., 1.5))

        ax = figw.add_subplot(gsw[j, i])
        for k in range(len(keylist)):
            key = keylist[k]
            if key == 'findpeak' or key == 'threshold' or key == 'fftrans':
                continue
            wdistlist = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
            # ax.plot(wdistlist['mu'] + jitter[key], wdistlist['wdist'], c=color[key], label='$'+label[key]+'$', marker='^')
            yerr = np.vstack([wdistlist['wdist'][:, 1] - wdistlist['wdist'][:, 0], wdistlist['wdist'][:, 2] - wdistlist['wdist'][:, 1]])
            ax.errorbar(wdistlist['mu'] + jitter[key], wdistlist['wdist'][:, 1], yerr=yerr, label='$'+label[key]+'$', c=color[key], marker='^')
            ax.fill_between(wdistlist['mu'] + jitter[key], wdistlist['wdist'][:, 0], wdistlist['wdist'][:, 2], fc=color[key], alpha=0.1, color=None)
        ax.set_xlabel(r'$\mu$')
        ax.set_ylabel(r'$W-dist/\mathrm{ns}$')
        ax.set_title(fr'$\tau={tau}\mathrm{{ns}},\,\sigma={sigma}\mathrm{{ns}}$')
        ax.set_ylim(0, lim['wdist'][i, j])
        # ax.set_yscale('log')
        ax.grid()
        if i == len(Sigma) - 1 and j == len(Tau) - 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1., 1.5))

        ax = figr.add_subplot(gsr[j, i])
        for k in range(len(keylist)):
            key = keylist[k]
            if key == 'findpeak' or key == 'threshold' or key == 'fftrans':
                continue
            rsslist = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
            # ax.plot(rsslist['mu'] + jitter[key], rsslist['RSS'], c=color[key], label='$'+label[key]+'$', marker='^')
            yerr = np.vstack([rsslist['RSS'][:, 1] - rsslist['RSS'][:, 0], rsslist['RSS'][:, 2] - rsslist['RSS'][:, 1]])
            ax.errorbar(rsslist['mu'] + jitter[key], rsslist['RSS'][:, 1], yerr=yerr, label='$'+label[key]+'$', c=color[key], marker='^')
            ax.fill_between(rsslist['mu'] + jitter[key], rsslist['RSS'][:, 0], rsslist['RSS'][:, 2], fc=color[key], alpha=0.1, color=None)
        ax.set_xlabel(r'$\mu$')
        ax.set_ylabel(r'$RSS/\mathrm{mV}^{2}$')
        ax.set_title(fr'$\tau={tau}\mathrm{{ns}},\,\sigma={sigma}\mathrm{{ns}}$')
        ax.set_ylim(0, lim['rss'][i, j])
        # ax.set_yscale('log')
        ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
        ax.grid()
        if i == len(Sigma) - 1 and j == len(Tau) - 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1., 1.5))
figd.savefig('Note/figures/vs-delta.pgf')
figd.savefig('Note/figures/vs-delta.pdf')
plt.close(figd)
figdd.savefig('Note/figures/vs-deltamethodsdiv.pgf')
figdd.savefig('Note/figures/vs-deltamethodsdiv.pdf')
plt.close(figdd)
figw.savefig('Note/figures/vs-wdist.pgf')
figw.savefig('Note/figures/vs-wdist.pdf')
plt.close(figw)
figr.savefig('Note/figures/vs-rss.pgf')
figr.savefig('Note/figures/vs-rss.pdf')
plt.close(figr)

fig = plt.figure(figsize=(len(Sigma) * 6, len(Tau) * 4))
gs = gridspec.GridSpec(len(Tau), len(Sigma), figure=fig, left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.2, hspace=0.3)
for sigma, i in zip(Sigma, list(range(len(Sigma)))):
    for tau, j in zip(Tau, list(range(len(Tau)))):
        ax = fig.add_subplot(gs[j, i])
        stdlist = mts['lucyddm'][(mts['lucyddm']['tau'] == tau) & (mts['lucyddm']['sigma'] == sigma)]
        alpha = 0.05
        yerr = stdlist['stdtruth'] / stdlist['std1sttruth'] / np.sqrt(stdlist['N'])
        ax.errorbar(stdlist['mu'], stdlist['stdtruth'] / stdlist['std1sttruth'], yerr=yerr, label='$\delta_'+deltalabel['tru']+'/\delta_'+deltalabel['1st']+'$', marker='^')
        ax.set_xlabel(r'$\mu$')
        ax.set_ylabel(r'$\mathrm{ratio}$')
        ax.set_title(fr'$\tau={tau}\mathrm{{ns}},\,\sigma={sigma}\mathrm{{ns}}$')
        ax.set_ylim(0.3, 1.1)
        ax.grid()
        ax.legend(loc='lower left')
fig.savefig('Note/figures/vs-deltadiv.pgf')
fig.savefig('Note/figures/vs-deltadiv.pdf')
plt.close(fig)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot()
tau = 20
sigma = 10
stdlist = mts['lucyddm'][(mts['lucyddm']['tau'] == tau) & (mts['lucyddm']['sigma'] == sigma)]
alpha = 0.05
yerr = stdlist['stdtruth'] / stdlist['std1sttruth'] / np.sqrt(stdlist['N'])
ax.errorbar(stdlist['mu'], stdlist['stdtruth'] / stdlist['std1sttruth'], yerr=yerr, label='$\delta_'+deltalabel['tru']+'/\delta_'+deltalabel['1st']+'$', marker='^')
ax.set_xlabel(r'$\mu$')
ax.set_ylabel(r'$\mathrm{ratio}$')
ax.set_title(fr'$\tau={tau}\mathrm{{ns}},\,\sigma={sigma}\mathrm{{ns}}$')
ax.set_ylim(0.3, 1.1)
ax.grid()
ax.legend(loc='lower left')
fig.savefig('Note/figures/2010vs-deltadiv.pgf')
fig.savefig('Note/figures/2010vs-deltadiv.pdf')

x = np.arange(0, len(keylist))
tau = 20
sigma = 10
wdist = np.vstack([mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]['wdist'].mean(axis=0) for key in keylist])
dy = np.vstack([wdist[:, 1] - wdist[:, 0], wdist[:, 2] - wdist[:, 1]])
fig = plt.figure(figsize=(12, 6))
# fig.tight_layout()
ax = fig.add_subplot(111)
ax.bar(x, wdist[:, 1], color='b')
ax.set_ylim(0, math.ceil(wdist[~np.isnan(wdist)].max() + 0.5))
ax.set_ylabel(r'$\mathrm{Wasserstein}\ \mathrm{Distance}/\mathrm{ns}$')
ax.set_xticks(x)
ax.set_xticklabels(['$'+label[key]+'$' for key in keylist])
ax.errorbar(x, wdist[:, 1], yerr=dy, fmt='o', ecolor='r', color='b', elinewidth=2, capsize=3)
ax.set_title(fr'$\tau={tau}\mathrm{{ns}},\,\sigma={sigma}\mathrm{{ns}}$')
fig.savefig('Note/figures/summarycharge.pgf')
fig.savefig('Note/figures/summarycharge.pdf')
fig.clf()
plt.close(fig)

for key in mts.keys():
    print(key.rjust(10) + ' stdchargesuccess mean = {:.04%}'.format((mts[key]['stdchargesuccess'] / mts[key]['N']).mean()))
    print(key.rjust(10) + '  stdchargesuccess min = {:.04%}'.format((mts[key]['stdchargesuccess'] / mts[key]['N']).min()))
for m in ['mcmc', 'fbmp']:
    print((m + 'wave').rjust(10) + '   stdwavesuccess mean = {:.04%}'.format((mts[m]['stdwavesuccess'] / mts[m]['N']).mean()))
    print((m + 'wave').rjust(10) + '    stdwavesuccess min = {:.04%}'.format((mts[m]['stdwavesuccess'] / mts[m]['N']).min()))