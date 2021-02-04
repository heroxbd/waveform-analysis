# -*- coding: utf-8 -*-

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
psr.add_argument('-o', dest='opt', type=str, help='output file')
psr.add_argument('--folder', nargs='+', type=str, help='folder of solution file')
psr.add_argument('--conf', type=str, help='configuration of tau & sigma')
args = psr.parse_args()

with open(args.conf) as f:
    f_csv = csv.reader(f, delimiter=' ')
    Tau = next(f_csv)
    Tau = [int(i) for i in Tau]
    Sigma = next(f_csv)
    Sigma = [int(i) for i in Sigma]

filelist = os.listdir(args.folder[0])
filelist = [f for f in filelist if f[0] != '.' and os.path.splitext(f)[-1] == '.h5']
numbers = [[float(i) for i in f[:-3].split('-')] for f in filelist]
stype = np.dtype([('mu', np.float), ('tau', np.float), ('sigma', np.float), ('n', np.uint), ('std1sttruth', np.float), ('stdtruth', np.float), ('stdcharge', np.float), ('stdwave', np.float), ('N', np.uint)])
mts = np.zeros(len(numbers), dtype=stype)
mts['mu'] = np.array([i[0] for i in numbers])
mts['tau'] = np.array([i[1] for i in numbers])
mts['sigma'] = np.array([i[2] for i in numbers])
mts['n'] = np.arange(len(numbers))
mts['N'] = np.nan
mts['std1sttruth'] = np.nan
mts['stdtruth'] = np.nan
mts['stdcharge'] = np.nan
mts['stdwave'] = np.nan
mts = np.sort(mts, kind='stable', order=['mu', 'tau', 'sigma'])

pdf = PdfPages(args.opt)
for i in range(len(mts)):
    f = filelist[mts[i]['n']]
    mu = mts[i]['mu']
    tau = mts[i]['tau']
    sigma = mts[i]['sigma']
    try:
        with h5py.File(os.path.join(args.folder[0], f), 'r', libver='latest', swmr=True) as soluf, h5py.File(os.path.join(args.folder[1], f), 'r', libver='latest', swmr=True) as wavef:
            time = soluf['starttime'][:]
            method = soluf['starttime'].attrs['Method']
            start = wavef['SimTruth/T'][:]
        mts[i]['N'] = len(start)
        mts[i]['std1sttruth'] = np.std(time['ts1sttruth'] - start['T0'], ddof=-1)
        mts[i]['stdtruth'] = np.std(time['tstruth'] - start['T0'], ddof=-1)
        mts[i]['stdcharge'] = np.std(time['tscharge'] - start['T0'], ddof=-1)
        mts[i]['stdwave'] = np.std(time['tswave'] - start['T0'], ddof=-1)
    except:
        pass

dhigh = np.array([np.max(mts['std1sttruth']), np.max(mts['stdtruth']), np.max(mts['stdcharge']), np.max(mts['stdwave'])])
dhigh = np.max(dhigh[~np.isnan(dhigh)]) * 1.05

def draw(tau, sigma, pdf):
    stdlist = mts[(mts['tau'] == tau) & (mts['sigma'] == sigma)]
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 1, figure=fig, left=0.15, right=0.95, top=0.9, bottom=0.1, wspace=0.2, hspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(stdlist['mu'], stdlist['std1sttruth'], label=r'$\delta_{1sttru}$', marker='^')
    ax.plot(stdlist['mu'], stdlist['stdtruth'], label=r'$\delta_{tru}$', marker='^')
    ax.plot(stdlist['mu'], stdlist['stdcharge'], label=r'$\delta_{cha}$', marker='^')
    if np.all(np.isnan(stdlist['stdwave'])):
        ax.plot(stdlist['mu'], stdlist['stdwave'], label=r'$\delta_{wave}$', marker='^')
    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'$\delta/\mathrm{{ns}}$')
    ax.set_title(fr'$\tau={tau}\mathrm{{ns}},\,\sigma={sigma}\mathrm{{ns}}$')
    ax.set_ylim(0, dhigh)
    ax.grid()
    ax.legend().set_zorder(1)
    pdf.savefig(fig)
    plt.close(fig)

tausigma = np.unique(np.vstack([mts['tau'], mts['sigma']]).T, axis=0)
for i in range(len(tausigma)):
    tau = tausigma[i][0]
    sigma = tausigma[i][1]
    draw(tau, sigma, pdf)
pdf.close()

if method == 'lucyddm':
    fig = plt.figure(figsize=(8, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, left=0.1, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.3)
    for sigma, i in zip(Sigma, [0, 1]):
        for tau, j in zip(Tau, [0, 1, 2]):
            ax = fig.add_subplot(gs[j, i])
            stdlist = mts[(mts['tau'] == tau) & (mts['sigma'] == sigma)]
            alpha = 0.05
            yerr1st = np.vstack([stdlist['std1sttruth']-np.sqrt(np.power(stdlist['std1sttruth'],2)*stdlist['N']/chi2.ppf(1-alpha/2, stdlist['N'])), np.sqrt(np.power(stdlist['std1sttruth'],2)*stdlist['N']/chi2.ppf(alpha/2, stdlist['N']))-stdlist['std1sttruth']])
            yerrall = np.vstack([stdlist['stdtruth']-np.sqrt(np.power(stdlist['stdtruth'],2)*stdlist['N']/chi2.ppf(1-alpha/2, stdlist['N'])), np.sqrt(np.power(stdlist['stdtruth'],2)*stdlist['N']/chi2.ppf(alpha/2, stdlist['N']))-stdlist['stdtruth']])
            yerr = np.vstack([stdlist['stdtruth'] / stdlist['std1sttruth'] - (stdlist['stdtruth'] - yerrall[0]) / (stdlist['std1sttruth'] + yerr1st[1]), (stdlist['stdtruth'] + yerrall[1]) / (stdlist['std1sttruth'] - yerr1st[0]) - stdlist['stdtruth'] / stdlist['std1sttruth']])
            ax.errorbar(stdlist['mu'], stdlist['stdtruth'] / stdlist['std1sttruth'], yerr=yerr, label=r'$\frac{\delta_{tru}}{\delta_{1sttru}}$', marker='^')
            ax.set_xlabel(r'$\mu$')
            ax.set_ylabel(r'$\mathrm{ratio}$')
            ax.set_title(fr'$\tau={tau}\mathrm{{ns}},\,\sigma={sigma}\mathrm{{ns}}$')
            ax.set_ylim(0.3, 1.05)
            ax.grid()
            ax.legend(loc='lower left').set_zorder(1)
    fig.savefig('Note/figures/vs-deltadiv.pgf')
    fig.savefig('Note/figures/vs-deltadiv.pdf')
    plt.close(fig)