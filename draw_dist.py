# -*- coding: utf-8 -*-

import sys
import h5py
import argparse

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', help='output file')
psr.add_argument('ipt', help='input file')
psr.add_argument('--mod', type=str, help='mode of weight', choices=['PEnum', 'Charge'])
psr.add_argument('-p', dest='pri', action='store_false', help='print bool', default=True)
args = psr.parse_args()
mode = args.mod
if mode == 'PEnum':
    extradist = 'pdist'
    pecount = 'TotalPEnum'
    extradistlabel = ['P-dist', r'$P-dist/\mathrm{1}$']
elif mode == 'Charge':
    extradist = 'chargediff'
    pecount = 'TotalPEpos'
    extradistlabel = ['Charge-diff', r'$Charge-diff/\mathrm{mV}\cdot\mathrm{ns}$']
if args.pri:
    sys.stdout = None

import csv
import numpy as np
from scipy import stats
from tqdm import tqdm
import matplotlib
matplotlib.use('pgf')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
import wf_func as wff

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.rcParams['figure.figsize'] = (12, 8)

def my_cmap():
    plasma = cm.get_cmap('plasma', 65536)
    newcolors = plasma(np.linspace(0, 1, 65536))
    white = np.array([65535/65536, 65535/65536, 65535/65536, 1])
    newcolors[:1, :] = white
    newcmp = ListedColormap(newcolors)
    return newcmp

mycmp = my_cmap()
with h5py.File(args.ipt, 'r', libver='latest', swmr=True) as distfile:
    dt = distfile['Record'][:]
    method = distfile['Record'].attrs['Method']

pdf = PdfPages(args.opt)
N = np.percentile(dt['wdist'], 95)
M = 800

fig = plt.figure()
ax = fig.add_subplot(111)
c, t = np.unique(dt[pecount], return_counts=True)
ax.bar(c, t)
ax.set_xlabel(pecount)
ax.set_ylabel('Count')
ax.set_yscale('log')
fig.suptitle(pecount + ' summary')
pdf.savefig(fig)
plt.close(fig)

penum = np.unique(dt[pecount])
l = min(50, penum.max())
wdist_stats = np.zeros((l, 6))
edist_stats = np.zeros((l, 6))
for i in tqdm(range(l), disable=args.pri):
    if i+1 in penum:
        dtwpi = dt['wdist'][dt[pecount] == i+1]
        dtepi = dt['RSS'][dt[pecount] == i+1]
        wdist_stats[i, 0] = np.median(dtwpi)
        wdist_stats[i, 1] = np.median(np.abs(dtwpi - np.median(dtwpi)))
        wdist_stats[i, 2] = np.mean(dtwpi)
        wdist_stats[i, 3] = np.std(dtwpi)
        wdist_stats[i, 4] = np.percentile(dtwpi, 10)
        wdist_stats[i, 5] = np.percentile(dtwpi, 90)
        edist_stats[i, 0] = np.median(dtepi)
        edist_stats[i, 1] = np.median(np.abs(dtepi - np.median(dtepi)))
        edist_stats[i, 2] = np.mean(dtepi)
        edist_stats[i, 3] = np.std(dtepi)
        edist_stats[i, 4] = np.percentile(dtepi, 10)
        edist_stats[i, 5] = np.percentile(dtepi, 90)
        rss = dt['RSS'][dt[pecount] == i+1]
        rss_recon = dt['RSS_recon'][dt[pecount] == i+1]
        rss_truth = dt['RSS_truth'][dt[pecount] == i+1]
        fig = plt.figure()
        gs = gridspec.GridSpec(2, 2, figure=fig, left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.15, hspace=0.2)
        ax0 = fig.add_subplot(gs[0, 0])
        n = max(np.percentile(dtwpi, 95), N)
        ax0.hist(dtwpi[dtwpi < n], bins=200)
        a = (dtwpi < n).sum()
        b = len(dtwpi)
        ax0.set_title('count {}(<{:.2f}ns)/{}={:.2f}'.format(a, n, b, a/b))
        ax0.set_xlabel(r'$W-dist/\mathrm{ns}$')
        ax1 = fig.add_subplot(gs[0, 1])
        r1 = np.percentile(rss, 0)
        r2 = np.percentile(rss, 98)
        ax1.hist(rss[(rss > r1) & (rss < r2)], bins=200, density=1)
        ax1.set_xlabel(r'$RSS/\mathrm{mV}^{2}$' + ', within ({:.2f}, {:.2f})'.format(r1, r2))
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(dt[extradist][dt[pecount] == i+1], bins=200)
        ax2.set_xlabel(extradistlabel[1])
        ax3 = fig.add_subplot(gs[1, 1])
        deltarss = rss_recon - rss_truth
        r1 = np.percentile(deltarss, 0)
        r2 = np.percentile(deltarss, 98)
        ax3.hist(deltarss[(deltarss > r1) & (deltarss < r2)], bins=200, density=1)
        ax3.set_xlabel(r'$RSS_{recon} - RSS_{truth}/\mathrm{mV}^{2}$' + ', within ({:.2f}, {:.2f})'.format(r1, r2))
        fig.suptitle(args.ipt.split('/')[-1] + ' ' + pecount + '={:.0f}'.format(i+1))
        pdf.savefig(fig)
        plt.close(fig)
    else:
        wdist_stats[i, :] = np.nan
        edist_stats[i, :] = np.nan

a = (dt['wdist'] < N).sum()
b = (dt['RSS'] < M).sum()
L = len(dt['wdist'])
fig = plt.figure()
gs = gridspec.GridSpec(2, 2, figure=fig, left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.2, hspace=0.3)
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(dt['wdist'][dt['wdist']<N], bins=100, density=1)
ax1.set_title('count {}(Wd<{:.2f}ns)/{}={:.2f}'.format(a, N, L, a/L))
ax1.set_xlabel(r'$W-dist/\mathrm{ns}$')
ax1.set_ylabel(r'$Normalized Count$')
ax2 = fig.add_subplot(gs[1, 0])
ax2.hist(dt['RSS'][np.abs(dt['RSS']) < M], bins=100, density=1)
ax1.set_ylabel(r'$Normalized Count$')
ax2.set_title('count {}(Rss<{}mV^2)/{}={:.2f}'.format(b, M, L, b/L))
ax2.set_xlabel(r'$RSS/\mathrm{mV}^{2}$')
ax3 = fig.add_subplot(gs[:, 1])
vali = np.logical_and(np.abs(dt['RSS'])<M, dt['wdist']<N)
h2 = ax3.hist2d(dt['wdist'][vali], dt['RSS'][vali], bins=(100, 100), cmap=mycmp)
fig.colorbar(h2[3], ax=ax3, aspect=50)
ax3.set_xlabel('$W-dist/\mathrm{ns}$')
ax3.set_ylabel(r'$RSS/\mathrm{mV}^{2}$')
ax3.set_title('W-dist&RSS hist, Wd<{:.2f}ns, Rss<{}mV^2,'.format(N, M))
fig.suptitle(args.ipt.split('/')[-1] + ' Dist stats, method = ' + str(method))
pdf.savefig(fig)
plt.close(fig)

fig = plt.figure()
gs = gridspec.GridSpec(1, 2, figure=fig, left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.2, hspace=0.2)
ax1 = fig.add_subplot(gs[0, 0])
ey1 = np.vstack([wdist_stats[:, 0]-wdist_stats[:, 4], wdist_stats[:, 5]-wdist_stats[:, 0]])
ax1.errorbar(np.arange(l), wdist_stats[:, 0], yerr=ey1, label=r'$W-dist^{90\%}_{10\%}$')
ax1.set_xlabel(pecount)
ax1.set_ylabel(r'$W-dist/\mathrm{ns}$')
ax1.set_title('W-dist vs ' + pecount + ' stats')
ax1.legend()
ax2 = fig.add_subplot(gs[0, 1])
ey2 = np.vstack([edist_stats[:, 0]-edist_stats[:, 4], edist_stats[:, 5]-edist_stats[:, 0]])
ax2.errorbar(np.arange(l), edist_stats[:, 0], yerr=ey2, label=r'$RSS^{90\%}_{10\%}$')
ax2.set_xlabel(pecount)
ax2.set_ylabel(r'$RSS/\mathrm{mV}^{2}$')
ax2.set_title('RSS vs ' + pecount + ' stats')
ax2.legend()
fig.suptitle(args.ipt.split('/')[-1] + ' Dist stats, method = ' + str(method))
pdf.savefig(fig)
plt.close(fig)

pdf.close()