import os
import sys
import re
import time
import math
import csv
import argparse

import h5py
import numpy as np
from scipy import stats
from scipy.stats import chi2
from tqdm import tqdm
import matplotlib
matplotlib.use('pgf')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec

psr = argparse.ArgumentParser()
psr.add_argument('--conf', type=str, help='configuration of tau & sigma')
args = psr.parse_args()

with open(args.conf) as f:
    f_csv = csv.reader(f, delimiter=' ')
    Tau = next(f_csv)
    Tau = [int(i) for i in Tau]
    Sigma = next(f_csv)
    Sigma = [int(i) for i in Sigma]

filelist = os.listdir('result/lucyddm/solu')
filelist = [f for f in filelist if f[0] != '.' and os.path.splitext(f)[-1] == '.h5']
numbers = [[float(i) for i in f[:-3].split('-')] for f in filelist]
stype = np.dtype([('mu', np.float64), ('tau', np.float64), ('sigma', np.float64), ('n', np.uint), ('std1sttruth', np.float64), ('stdtruth', np.float64), ('stdcharge', np.float64), ('stdwave', np.float64), ('wdist', np.float64), ('RSS', np.float64), ('N', np.uint)])
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

# mts = {'findpeak':mtsi.copy(), 'threshold':mtsi.copy(), 'fftrans':mtsi.copy(), 'lucyddm':mtsi.copy(), 'xiaopeip':mtsi.copy(), 'mcmcrec':mtsi.copy()}
mts = {'fftrans':mtsi.copy(), 'lucyddm':mtsi.copy(), 'xiaopeip':mtsi.copy(), 'mcmcrec':mtsi.copy()}

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
            mts[key][i]['N'] = len(start)
            mts[key][i]['std1sttruth'] = np.std(time['ts1sttruth'] - start['T0'], ddof=-1)
            mts[key][i]['stdtruth'] = np.std(time['tstruth'] - start['T0'], ddof=-1)
            mts[key][i]['stdcharge'] = np.std(time['tscharge'] - start['T0'], ddof=-1)
            mts[key][i]['stdwave'] = np.std(time['tswave'] - start['T0'], ddof=-1)
            mts[key][i]['wdist'] = record['wdist'].mean()
            mts[key][i]['RSS'] = record['RSS'].mean()
        except:
            pass

dhigh = np.array([[np.max(mts[key]['std1sttruth']), np.max(mts[key]['stdtruth']), np.max(mts[key]['stdcharge']), np.max(mts[key]['stdwave'])] for key in mts.keys()])
dhigh = np.max(dhigh[~np.isnan(dhigh)]) * 1.05
whigh = np.array([[np.max(mts[key]['wdist'])] for key in mts.keys()])
whigh = np.max(whigh[~np.isnan(whigh)]) * 1.05
rhigh = np.array([[np.max(mts[key]['RSS'])] for key in mts.keys()])
rhigh = np.max(rhigh[~np.isnan(rhigh)]) * 1.05

label = {'1st':r'$_{1st}$', 'tru':r'$_{tru}$', 'fftrans':r'$_{fftrans}$', 'lucyddm':r'$_{lucyddm}$', 'xiaopeip':r'$_{xiaopeip}$', 'mcmcrec':r'$_{mcmccha}$', 'wave':r'$_{mcmct0}$'}
color = {'1st':'b', 'tru':'k', 'fftrans':'m', 'lucyddm':'y', 'xiaopeip':'c', 'mcmcrec':'r', 'wave':'g'}

figd = plt.figure(figsize=(8, 12))
gsd = gridspec.GridSpec(3, 2, figure=figd, left=0.1, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.3)
figdd = plt.figure(figsize=(8, 12))
gsdd = gridspec.GridSpec(3, 2, figure=figd, left=0.1, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.3)
figw = plt.figure(figsize=(8, 12))
gsw = gridspec.GridSpec(3, 2, figure=figw, left=0.1, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.3)
figr = plt.figure(figsize=(8, 12))
gsr = gridspec.GridSpec(3, 2, figure=figr, left=0.1, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.3)
alpha = 0.05
low = np.array([[0.3, 0.5, 0.6], [0.3, 0.5, 0.6]])
for sigma, i in zip(Sigma, [0, 1]):
    for tau, j in zip(Tau, [0, 1, 2]):
        ax = figd.add_subplot(gsd[j, i])
        stdlist = mts['lucyddm'][(mts['lucyddm']['tau'] == tau) & (mts['lucyddm']['sigma'] == sigma)]
        yerr1st = np.vstack([stdlist['std1sttruth']-np.sqrt(np.power(stdlist['std1sttruth'],2)*stdlist['N']/chi2.ppf(1-alpha/2, stdlist['N'])), np.sqrt(np.power(stdlist['std1sttruth'],2)*stdlist['N']/chi2.ppf(alpha/2, stdlist['N']))-stdlist['std1sttruth']])
        yerrall = np.vstack([stdlist['stdtruth']-np.sqrt(np.power(stdlist['stdtruth'],2)*stdlist['N']/chi2.ppf(1-alpha/2, stdlist['N'])), np.sqrt(np.power(stdlist['stdtruth'],2)*stdlist['N']/chi2.ppf(alpha/2, stdlist['N']))-stdlist['stdtruth']])
        ax.errorbar(stdlist['mu'], stdlist['std1sttruth'], yerr=yerr1st, c=color['1st'], label=r'$\delta$'+label['1st'], marker='^')
        ax.errorbar(stdlist['mu'], stdlist['stdtruth'], yerr=yerrall, c=color['tru'], label=r'$\delta$'+label['tru'], marker='^')
        stdlistwav = mts['mcmcrec'][(mts['mcmcrec']['tau'] == tau) & (mts['mcmcrec']['sigma'] == sigma)]
        yerrwav = np.vstack([stdlistwav['stdwave']-np.sqrt(np.power(stdlistwav['stdwave'],2)*stdlistwav['N']/chi2.ppf(1-alpha/2, stdlistwav['N'])), np.sqrt(np.power(stdlistwav['stdwave'],2)*stdlistwav['N']/chi2.ppf(alpha/2, stdlistwav['N']))-stdlistwav['stdwave']])
        ax.errorbar(stdlistwav['mu'], stdlistwav['stdwave'], yerr=yerrwav, c=color['wave'], label=r'$\delta$'+label['wave'], marker='^')
        for key in mts.keys():
            stdlist = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
            yerrcha = np.vstack([stdlist['stdcharge']-np.sqrt(np.power(stdlist['stdcharge'],2)*stdlist['N']/chi2.ppf(1-alpha/2, stdlist['N'])), np.sqrt(np.power(stdlist['stdcharge'],2)*stdlist['N']/chi2.ppf(alpha/2, stdlist['N']))-stdlist['stdcharge']])
            ax.errorbar(stdlist['mu'], stdlist['stdcharge'], yerr=yerrcha, c=color[key], label=r'$\delta$'+label[key], marker='^')
        ax.set_xlabel(r'$\mu$')
        ax.set_ylabel(r'$\delta/\mathrm{ns}$')
        ax.set_title(fr'$\tau={tau}\mathrm{{ns}},\,\sigma={sigma}\mathrm{{ns}}$')
        # ax.set_ylim(0, dhigh)
        # ax.set_yscale('log')
        ax.grid()
        ax.legend(loc='upper right')

        ax = figdd.add_subplot(gsdd[j, i])
        stdlist = mts['lucyddm'][(mts['lucyddm']['tau'] == tau) & (mts['lucyddm']['sigma'] == sigma)]
        yerr1st = np.vstack([stdlist['std1sttruth']-np.sqrt(np.power(stdlist['std1sttruth'],2)*stdlist['N']/chi2.ppf(1-alpha/2, stdlist['N'])), np.sqrt(np.power(stdlist['std1sttruth'],2)*stdlist['N']/chi2.ppf(alpha/2, stdlist['N']))-stdlist['std1sttruth']])
        yerrall = np.vstack([stdlist['stdtruth']-np.sqrt(np.power(stdlist['stdtruth'],2)*stdlist['N']/chi2.ppf(1-alpha/2, stdlist['N'])), np.sqrt(np.power(stdlist['stdtruth'],2)*stdlist['N']/chi2.ppf(alpha/2, stdlist['N']))-stdlist['stdtruth']])
        yerrall = np.vstack([stdlist['stdtruth'] / stdlist['std1sttruth'] - (stdlist['stdtruth'] - yerrall[0]) / (stdlist['std1sttruth'] + yerr1st[1]), (stdlist['stdtruth'] + yerrall[1]) / (stdlist['std1sttruth'] - yerr1st[0]) - stdlist['stdtruth'] / stdlist['std1sttruth']])
        ax.errorbar(stdlist['mu'], stdlist['stdtruth'] / stdlist['std1sttruth'], yerr=yerrall, label=r'$\delta$'+label['tru']+r'$/\delta_{1st}$', c=color['tru'], marker='^')
        stdlistwav = mts['mcmcrec'][(mts['mcmcrec']['tau'] == tau) & (mts['mcmcrec']['sigma'] == sigma)]
        yerrwav = np.vstack([stdlistwav['stdwave']-np.sqrt(np.power(stdlistwav['stdwave'],2)*stdlistwav['N']/chi2.ppf(1-alpha/2, stdlistwav['N'])), np.sqrt(np.power(stdlistwav['stdwave'],2)*stdlistwav['N']/chi2.ppf(alpha/2, stdlistwav['N']))-stdlistwav['stdwave']])
        yerrwav = np.vstack([stdlistwav['stdwave'] / stdlist['std1sttruth'] - (stdlistwav['stdwave'] - yerrwav[0]) / (stdlist['std1sttruth'] + yerr1st[1]), (stdlistwav['stdwave'] + yerrwav[1]) / (stdlist['std1sttruth'] - yerr1st[0]) - stdlistwav['stdwave'] / stdlist['std1sttruth']])
        ax.errorbar(stdlistwav['mu'], stdlistwav['stdwave'] / stdlist['std1sttruth'], yerr=yerrwav, label=r'$\delta$'+label['wave']+r'$/\delta_{1st}$', c=color['wave'], marker='^')
        for key in mts.keys():
            stdlistkey = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
            yerrcha = np.vstack([stdlist['stdcharge']-np.sqrt(np.power(stdlist['stdcharge'],2)*stdlist['N']/chi2.ppf(1-alpha/2, stdlist['N'])), np.sqrt(np.power(stdlist['stdcharge'],2)*stdlist['N']/chi2.ppf(alpha/2, stdlist['N']))-stdlist['stdcharge']])
            yerr = np.vstack([stdlist['std1sttruth'] / stdlistkey['stdcharge'] - (stdlist['std1sttruth'] - yerrall[0]) / (stdlistkey['stdcharge'] + yerrcha[1]), (stdlist['std1sttruth'] + yerrall[1]) / (stdlistkey['stdcharge'] - yerrcha[0]) - stdlist['std1sttruth'] / stdlistkey['stdcharge']])
            ax.errorbar(stdlist['mu'], stdlistkey['stdcharge'] / stdlist['std1sttruth'], yerr=yerr, label=r'$\delta$'+label[key]+r'$/\delta_{{1st}}$', c=color[key], marker='^')
        ax.set_xlabel(r'$\mu$')
        ax.set_ylabel(r'$ratio$')
        ax.set_title(fr'$\tau={tau}\mathrm{{ns}},\,\sigma={sigma}\mathrm{{ns}}$')
        ax.set_ylim(low[i, j], 1.1)
        ax.grid()
        ax.legend(loc='lower left')

        ax = figw.add_subplot(gsw[j, i])
        for key in mts.keys():
            wdistlist = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
            ax.plot(wdistlist['mu'], wdistlist['wdist'], c=color[key], label=r'$W-dist$'+label[key], marker='^')
        ax.set_xlabel(r'$\mu$')
        ax.set_ylabel(r'$W-dist/\mathrm{ns}$')
        ax.set_title(fr'$\tau={tau}\mathrm{{ns}},\,\sigma={sigma}\mathrm{{ns}}$')
        # ax.set_ylim(0, whigh)
        # ax.set_yscale('log')
        ax.grid()
        ax.legend(loc='upper right')
        ax = figr.add_subplot(gsr[j, i])
        for key in mts.keys():
            rsslist = mts[key][(mts[key]['tau'] == tau) & (mts[key]['sigma'] == sigma)]
            ax.plot(rsslist['mu'], rsslist['RSS'], c=color[key], label=r'$RSS$'+label[key], marker='^')
        ax.set_xlabel(r'$\mu$')
        ax.set_ylabel(r'$RSS/\mathrm{mV}^{2}$')
        ax.set_title(fr'$\tau={tau}\mathrm{{ns}},\,\sigma={sigma}\mathrm{{ns}}$')
        # ax.set_ylim(0, rhigh)
        # ax.set_yscale('log')
        ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
        ax.grid()
        ax.legend(loc='upper left')
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

fig = plt.figure(figsize=(8, 12))
gs = gridspec.GridSpec(3, 2, figure=fig, left=0.1, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.3)
for sigma, i in zip(Sigma, [0, 1]):
    for tau, j in zip(Tau, [0, 1, 2]):
        ax = fig.add_subplot(gs[j, i])
        stdlist = mts['lucyddm'][(mts['lucyddm']['tau'] == tau) & (mts['lucyddm']['sigma'] == sigma)]
        alpha = 0.05
        yerr1st = np.vstack([stdlist['std1sttruth']-np.sqrt(np.power(stdlist['std1sttruth'],2)*stdlist['N']/chi2.ppf(1-alpha/2, stdlist['N'])), np.sqrt(np.power(stdlist['std1sttruth'],2)*stdlist['N']/chi2.ppf(alpha/2, stdlist['N']))-stdlist['std1sttruth']])
        yerrall = np.vstack([stdlist['stdtruth']-np.sqrt(np.power(stdlist['stdtruth'],2)*stdlist['N']/chi2.ppf(1-alpha/2, stdlist['N'])), np.sqrt(np.power(stdlist['stdtruth'],2)*stdlist['N']/chi2.ppf(alpha/2, stdlist['N']))-stdlist['stdtruth']])
        yerr = np.vstack([stdlist['stdtruth'] / stdlist['std1sttruth'] - (stdlist['stdtruth'] - yerrall[0]) / (stdlist['std1sttruth'] + yerr1st[1]), (stdlist['stdtruth'] + yerrall[1]) / (stdlist['std1sttruth'] - yerr1st[0]) - stdlist['stdtruth'] / stdlist['std1sttruth']])
        ax.errorbar(stdlist['mu'], stdlist['stdtruth'] / stdlist['std1sttruth'], yerr=yerr, label=r'$\delta_{tru}/\delta_{1st}$', marker='^')
        ax.set_xlabel(r'$\mu$')
        ax.set_ylabel(r'$\mathrm{ratio}$')
        ax.set_title(fr'$\tau={tau}\mathrm{{ns}},\,\sigma={sigma}\mathrm{{ns}}$')
        ax.set_ylim(0.3, 1.1)
        ax.grid()
        ax.legend(loc='lower left')
fig.savefig('Note/figures/vs-deltadiv.pgf')
fig.savefig('Note/figures/vs-deltadiv.pdf')
plt.close(fig)