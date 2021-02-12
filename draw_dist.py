import sys
import h5py
import argparse

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', help='output file')
psr.add_argument('ipt', help='input file')
psr.add_argument('--ref', type=str, nargs='+', help='reference file')
args = psr.parse_args()

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
M = np.percentile(dt['RSS'], 95)

fig = plt.figure()
ax = fig.add_subplot(111)
c, t = np.unique(dt['NPE'], return_counts=True)
ax.bar(c, t)
ax.set_xlabel(r'$N_{pe}$')
ax.set_ylabel('Count')
ax.set_yscale('log')
fig.suptitle(r'$N_{pe}$' + ' summary')
pdf.savefig(fig)
plt.close(fig)

penum = np.unique(dt['NPE'])
l = min(50, penum.max())
wdist_stats = np.zeros((l, 6))
edist_stats = np.zeros((l, 6))
for i in range(l):
    if i+1 in penum:
        dtwpi = dt['wdist'][dt['NPE'] == i+1]
        dtepi = dt['RSS'][dt['NPE'] == i+1]
        wdist_stats[i, 0] = np.median(dtwpi)
        wdist_stats[i, 1] = np.median(np.abs(dtwpi - np.median(dtwpi)))
        wdist_stats[i, 2] = np.mean(dtwpi)
        wdist_stats[i, 3] = np.std(dtwpi)
        wdist_stats[i, 4] = np.percentile(dtwpi, 5)
        wdist_stats[i, 5] = np.percentile(dtwpi, 95)
        edist_stats[i, 0] = np.median(dtepi)
        edist_stats[i, 1] = np.median(np.abs(dtepi - np.median(dtepi)))
        edist_stats[i, 2] = np.mean(dtepi)
        edist_stats[i, 3] = np.std(dtepi)
        edist_stats[i, 4] = np.percentile(dtepi, 5)
        edist_stats[i, 5] = np.percentile(dtepi, 95)
        rss = dt['RSS'][dt['NPE'] == i+1]
        rss_recon = dt['RSS_recon'][dt['NPE'] == i+1]
        rss_truth = dt['RSS_truth'][dt['NPE'] == i+1]
    else:
        wdist_stats[i, :] = np.nan
        edist_stats[i, :] = np.nan

a = (dt['wdist'] < N).sum()
b = (dt['RSS'] < M).sum()
L = len(dt['wdist'])
fig = plt.figure()
gs = gridspec.GridSpec(2, 2, figure=fig, left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.2, hspace=0.3)
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(dt['wdist'][dt['wdist'] < N], bins=50)
ax1.set_title('count {}(Wd<{:.02f}ns)/{}={:.02f}'.format(a, N, L, a/L))
ax1.set_xlabel(r'$W-dist/\mathrm{ns}$')
ax1.set_ylabel(r'$Count$')
# ax1.set_yscale('log')
ax2 = fig.add_subplot(gs[1, 0])
ax2.hist(dt['RSS'][np.abs(dt['RSS']) < M], bins=50)
ax2.set_ylabel(r'$Count$')
# ax2.set_yscale('log')
ax2.set_title('count {}(RSS<{:.02f}mV^2)/{}={:.02f}'.format(b, M, L, b/L))
ax2.set_xlabel(r'$RSS/\mathrm{mV}^{2}$')
# ax2.yaxis.get_major_formatter().set_powerlimits((0, 1))
ax3 = fig.add_subplot(gs[:, 1])
vali = np.logical_and(np.abs(dt['RSS'])<M, dt['wdist']<N)
h2 = ax3.hist2d(dt['wdist'][vali], dt['RSS'][vali], bins=(50, 50), cmap=mycmp)
fig.colorbar(h2[3], ax=ax3, aspect=50)
ax3.set_xlabel('$W-dist/\mathrm{ns}$')
ax3.set_ylabel(r'$RSS/\mathrm{mV}^{2}$')
ax3.set_title('W-dist&RSS hist, Wd<{:.02f}ns, RSS<{:.02f}mV^2,'.format(N, M))
# ax3.yaxis.get_major_formatter().set_powerlimits((0, 1))
fig.suptitle(args.ipt.split('/')[-1] + ' Dist stats, method = ' + str(method))
pdf.savefig(fig)
plt.close(fig)

fig = plt.figure()
gs = gridspec.GridSpec(1, 2, figure=fig, left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.2, hspace=0.2)
ax1 = fig.add_subplot(gs[0, 0])
ax1.boxplot(np.array([dt['wdist'][dt['NPE'] == i+1] for i in range(l)], dtype=object), sym='', patch_artist=True)
ax1.plot(np.arange(1, l + 1), wdist_stats[:, 0], label=r'$W-dist$')
ax1.set_xlabel(r'$N_{pe}$')
ax1.set_xlim(penum[0] - 0.5, penum[-1] + 0.5)
ax1.set_ylabel(r'$W-dist/\mathrm{ns}$')
ax1.set_title('W-dist vs ' + r'$N_{pe}$' + ' stats')
ax1.legend()
ax2 = fig.add_subplot(gs[0, 1])
ax2.boxplot(np.array([dt['RSS'][dt['NPE'] == i+1] for i in range(l)], dtype=object), sym='', patch_artist=True)
ax2.plot(np.arange(1, l + 1), edist_stats[:, 0], label=r'$RSS$')
ax2.set_xlabel(r'$N_{pe}$')
ax2.set_xlim(penum[0] - 0.5, penum[-1] + 0.5)
ax2.set_ylabel(r'$RSS/\mathrm{mV}^{2}$')
ax2.set_title('RSS vs ' + r'$N_{pe}$' + ' stats')
ax2.legend()
fig.suptitle(args.ipt.split('/')[-1] + ' Dist stats, method = ' + str(method))
pdf.savefig(fig)
plt.close(fig)

with h5py.File(args.ref[0], 'r', libver='latest', swmr=True) as wavef, h5py.File(args.ref[1], 'r', libver='latest', swmr=True) as soluf:
    start = wavef['SimTruth/T'][:]
    time = soluf['starttime'][:]

fig = plt.figure()
gs = gridspec.GridSpec(2, 2, figure=fig, left=0.1, right=0.95, top=0.9, bottom=0.1, wspace=0.2, hspace=0.3)

ax0 = fig.add_subplot(gs[0, 0])
ax0.hist(time['ts1sttruth'] - start['T0'], bins=100, label=r'$t_{1sttru} - t_{0}$')
ax0.set_xlabel(r'$t_{1sttru} - t_{0}/\mathrm{s}$')
ax0.set_ylabel(r'$Count$')
ax0.set_yscale('log')
ax0.legend()
s = np.std(time['ts1sttruth'] - start['T0'], ddof=-1)
ax0.set_title(fr'$\sigma_{{1sttru}}={s:.02f}$')

ax1 = fig.add_subplot(gs[0, 1])
ax1.hist(time['tstruth'] - start['T0'], bins=100, label=r'$t_{truth} - t_{0}$')
ax1.set_xlabel(r'$t_{truth} - t_{0}/\mathrm{s}$')
ax1.set_ylabel(r'$Count$')
ax1.set_yscale('log')
ax1.legend()
s = np.std(time['tstruth'] - start['T0'], ddof=-1)
ax1.set_title(fr'$\sigma_{{truth}}={s:.02f}$')

ax2 = fig.add_subplot(gs[1, 0])
ax2.hist(time['tscharge'] - start['T0'], bins=100, label=r'$t_{charge} - t_{0}$')
ax2.set_xlabel(r'$t_{charge} - t_{0}/\mathrm{s}$')
ax2.set_ylabel(r'$Count$')
ax2.set_yscale('log')
ax2.legend()
s = np.std(time['tscharge'] - start['T0'], ddof=-1)
ax2.set_title(fr'$\sigma_{{charge}}={s:.02f}$')

if not np.all(np.isnan(time['tswave'])):
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(time['tswave'] - start['T0'], bins=100, label=r'$t_{wave} - t_{0}$')
    ax3.set_xlabel(r'$t_{wave} - t_{0}/\mathrm{s}$')
    ax3.set_ylabel(r'$Count$')
    ax3.set_yscale('log')
    ax3.legend()
    s = np.std(time['tswave'] - start['T0'], ddof=-1)
    ax3.set_title(fr'$\sigma_{{wave}}={s:.02f}$')

pdf.savefig(fig)
plt.close(fig)

for i in range(l):
    if i+1 in penum:
        dtwpi = dt['wdist'][dt['NPE'] == i+1]
        rss = dt['RSS'][dt['NPE'] == i+1]
        rss_recon = dt['RSS_recon'][dt['NPE'] == i+1]
        rss_truth = dt['RSS_truth'][dt['NPE'] == i+1]
        charged = dt['chargediff'][dt['NPE'] == i+1]
        deltarss = rss_recon - rss_truth
        fig = plt.figure()
        gs = gridspec.GridSpec(2, 2, figure=fig, left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.15, hspace=0.2)
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.hist(dtwpi, bins=np.arange(0, dtwpi.max()+0.01, 0.01))
        # n = max(np.percentile(dtwpi, 95), N)
        # ax0.hist(dtwpi[dtwpi < n], bins=50)
        # a = (dtwpi < n).sum()
        # b = len(dtwpi)
        # ax0.set_title('count {}(<{:.02f}ns)/{}={:.02f}'.format(a, n, b, a/b))
        ax0.set_xlabel(r'$W-dist/\mathrm{ns}$')
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.hist(rss, bins=np.arange(0, rss.max()+50, 50))
        ax1.set_xlabel(r'$RSS/\mathrm{mV}^{2}$')
        # r1 = np.percentile(rss, 0)
        # r2 = np.percentile(rss, 98)
        # ax1.hist(rss[(rss > r1) & (rss < r2)], bins=50)
        # ax1.set_xlabel(r'$RSS/\mathrm{mV}^{2}$' + ', within ({:.02f}, {:.02f})'.format(r1, r2))
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(charged, bins=np.arange(0, charged.max()+2, 2))
        ax2.set_xlabel(r'$Charge-diff/\mathrm{mV}\cdot\mathrm{ns}$')
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.hist(deltarss, bins=np.arange(0, deltarss.max()+50, 50))
        ax3.set_xlabel(r'$RSS_{recon} - RSS_{truth}/\mathrm{mV}^{2}$')
        # r1 = np.percentile(deltarss, 0)
        # r2 = np.percentile(deltarss, 98)
        # ax3.hist(deltarss[(deltarss > r1) & (deltarss < r2)], bins=50)
        # ax3.set_xlabel(r'$RSS_{recon} - RSS_{truth}/\mathrm{mV}^{2}$' + ', within ({:.02f}, {:.02f})'.format(r1, r2))
        fig.suptitle(args.ipt.split('/')[-1] + ' ' + r'$N_{pe}$' + ' = {:.0f}'.format(i+1) + ' ' + 'count = {}'.format(sum(dt['NPE'] == i+1)))
        pdf.savefig(fig)
        plt.close(fig)

pdf.close()