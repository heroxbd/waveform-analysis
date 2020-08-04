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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['lines.markersize'] = 2
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['mathtext.fontset'] = 'cm'

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
    M = 500
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
    wdist_stats = np.zeros((l, 4))
    edist_stats = np.zeros((l, 4))
    for i in tqdm(range(l), disable=args.pri):
        if i+1 in penum:
            dtwpi = dt['wdist'][dt[pecount] == i+1]
            dtepi = dt[extradist][dt[pecount] == i+1]
            wdist_stats[i, 0] = np.median(dtwpi)
            wdist_stats[i, 1] = np.median(np.absolute(dtwpi - np.median(dtwpi)))
            wdist_stats[i, 2] = np.mean(dtwpi)
            wdist_stats[i, 3] = np.std(dtwpi)
            edist_stats[i, 0] = np.median(dtepi)
            edist_stats[i, 1] = np.median(np.absolute(dtepi - np.median(dtepi)))
            edist_stats[i, 2] = np.mean(dtepi)
            edist_stats[i, 3] = np.std(dtepi)
            rss_recon = dt['RSS_recon'][dt[pecount] == i+1]
            rss_truth = dt['RSS_truth'][dt[pecount] == i+1]
            plt.rcParams['figure.figsize'] = (12, 6)
            fig = plt.figure()
            gs = gridspec.GridSpec(2, 2, figure=fig, left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.15, hspace=0.2)
            ax1 = fig.add_subplot(gs[0, 0])
            n = max(np.percentile(dtwpi, 95), N)
            ax1.hist(dtwpi[dtwpi < n], bins=200)
            a = (dtwpi < n).sum()
            b = len(dtwpi)
            ax1.set_title('count {}(<{:.2f}ns)/{}={:.2f}'.format(a, n, b, a/b))
            ax1.set_xlabel('$W-dist/\mathrm{ns}$')
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.hist(dtepi, bins=100)
            ax2.set_xlabel(extradistlabel[1])
            ax3 = fig.add_subplot(gs[1, 1])
            deltarss = rss_recon - rss_truth
            r1 = np.percentile(deltarss, 0)
            r2 = np.percentile(deltarss, 98)
            ax3.hist(deltarss[(deltarss > r1) & (deltarss < r2)], bins=200, density=1)
            ax3.set_xlabel('$\mathrm{RSS}_{recon} - \mathrm{RSS}_{truth}/\mathrm{mV}^{2}$' + ', within ({:.2f}, {:.2f})'.format(r1, r2))
            fig.suptitle(args.ipt.split('/')[-1] + ' ' + pecount + '={:.0f}'.format(i+1))
            pdf.savefig(fig)
            plt.close(fig)
        else:
            wdist_stats[i, :] = np.nan
            edist_stats[i, :] = np.nan

    a = (dt['wdist'] < N).sum()
    b = (np.abs(dt[extradist]) < M).sum()
    l = len(dt['wdist'])
    if mode == 'PEnum':
        extradisttitle = None
        sumtitle = 'W&'+extradistlabel[0]+' hist,Wd<{:.2f}ns,'.format(N)+'flawed'
    elif mode == 'Charge':
        extradisttitle = 'count {}(|Cd|<{}mV*ns)/{}={:.2f}'.format(b, M, l, b/l)
        sumtitle = 'W-dist&'+extradistlabel[0]+' hist,Wd<{:.2f}ns,'.format(N)+'|Cd|<{}mV*ns,'.format(M)+'flawed'
    plt.rcParams['figure.figsize'] = (12, 6)
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 2, figure=fig, left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.2, hspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(dt['wdist'][dt['wdist']<N], bins=100, density=1)
    ax1.set_title('count {}(Wd<{:.2f}ns)/{}={:.2f}'.format(a, N, l, a/l))
    ax1.set_xlabel('$W-dist/\mathrm{ns}$')
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(dt[extradist][np.abs(dt[extradist]) < M], bins=100, density=1)
    ax2.set_title(extradisttitle)
    ax2.set_xlabel(extradistlabel[1])
    ax3 = fig.add_subplot(gs[:, 1])
    vali = np.logical_and(np.logical_and(np.abs(dt[extradist])<M, dt['wdist']<N),np.logical_and(dt[extradist]!=0, dt['wdist']!=0))
    h2 = ax3.hist2d(dt['wdist'][vali], dt[extradist][vali], bins=(100, 100), cmap=mycmp)
    fig.colorbar(h2[3], ax=ax3, aspect=50)
    ax3.set_xlabel('$W-dist/\mathrm{ns}$')
    ax3.set_ylabel(extradistlabel[1])
    ax3.set_title(sumtitle)
    fig.suptitle(args.ipt.split('/')[-1] + ' Dist stats, method = ' + str(method))
    pdf.savefig(fig)
    plt.close(fig)

    plt.rcParams['figure.figsize'] = (12, 6)
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2, figure=fig, left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.2, hspace=0.2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(wdist_stats[:, 0], c='C0', label='W median')
    ax1.plot(wdist_stats[:, 0] + wdist_stats[:, 1], c='C1', label='W median + mad')
    ax1.plot(wdist_stats[:, 0] - wdist_stats[:, 1], c='C1', label='W median - mad')
    ax1.plot(wdist_stats[:, 2], c='C2', label='W mean')
    ax1.plot(wdist_stats[:, 2] + wdist_stats[:, 3], c='C3', label='W mean + std')
    ax1.plot(wdist_stats[:, 2] - wdist_stats[:, 3], c='C3', label='W mean - std')
    ax1.set_xlabel(pecount)
    ax1.set_ylabel('$W-dist/\mathrm{ns}$')
    ax1.set_title('W-dist vs ' + pecount + ' stats')
    ax1.legend()
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(edist_stats[:, 0], c='C0', label=extradistlabel[0][0] + ' median')
    ax2.plot(edist_stats[:, 0] + edist_stats[:, 1], c='C1', label=extradistlabel[0][0] + ' median + mad')
    ax2.plot(edist_stats[:, 0] - edist_stats[:, 1], c='C1', label=extradistlabel[0][0] + ' median - mad')
    ax2.plot(edist_stats[:, 2], c='C2', label=extradistlabel[0][0] + ' mean')
    ax2.plot(edist_stats[:, 2] + edist_stats[:, 3], c='C3', label=extradistlabel[0][0] + ' mean + std')
    ax2.plot(edist_stats[:, 2] - edist_stats[:, 3], c='C3', label=extradistlabel[0][0] + ' mean - std')
    ax2.set_xlabel(pecount)
    ax2.set_ylabel(extradistlabel[1])
    ax2.set_title(extradistlabel[0] + ' vs ' + pecount + ' stats')
    ax2.legend()
    fig.suptitle(args.ipt.split('/')[-1] + ' Dist stats, method = ' + str(method))
    pdf.savefig(fig)
    plt.close(fig)

    pdf.close()
