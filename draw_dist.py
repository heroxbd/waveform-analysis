# -*- coding: utf-8 -*-

import sys
import h5py
import argparse

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', help='output file')
psr.add_argument('ipt', help='input file')
psr.add_argument('-p', dest='pri', action='store_false', help='print bool', default=True)
args = psr.parse_args()

import csv
import numpy as np
from scipy import stats
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

if args.pri:
    sys.stdout = None

def my_cmap():
    plasma = cm.get_cmap('plasma', 65536)
    newcolors = plasma(np.linspace(0, 1, 65536))
    white = np.array([65535/65536, 65535/65536, 65535/65536, 1])
    newcolors[:1, :] = white
    newcmp = ListedColormap(newcolors)
    return newcmp

if __name__ == '__main__':
    mycmp = my_cmap()
    with h5py.File(args.ipt, 'r', libver='latest', swmr=True) as distfile:
        dt = distfile['Record']
        method = dt.attrs['Method']
        pdf = PdfPages(args.opt)
        N = int(np.percentile(dt['wdist'], 90)+1)

        penum = np.unique(dt['PEnum'])
        l = min(50, penum.max())
        wdist_stats = np.zeros((l, 4))
        pdist_stats = np.zeros((l, 4))
        for i in np.arange(l):
            if i+1 in penum:
                dtwpi = dt['wdist'][dt['PEnum'] == i+1]
                dtppi = dt['pdist'][dt['PEnum'] == i+1]
                wdist_stats[i, 0] = np.median(dtwpi)
                wdist_stats[i, 1] = np.median(np.absolute(dtwpi - np.median(dtwpi)))
                wdist_stats[i, 2] = np.mean(dtwpi)
                wdist_stats[i, 3] = np.std(dtwpi)
                pdist_stats[i, 0] = np.median(dtppi)
                pdist_stats[i, 1] = np.median(np.absolute(dtppi - np.median(dtppi)))
                pdist_stats[i, 2] = np.mean(dtppi)
                pdist_stats[i, 3] = np.std(dtppi)
                pediff = dt['PEdiff'][dt['PEnum'] == i+1]
                rss_recon = dt['RSS_recon'][dt['PEnum'] == i+1]
                rss_truth = dt['RSS_truth'][dt['PEnum'] == i+1]
                plt.rcParams['figure.figsize'] = (12, 6)
                fig = plt.figure()
                gs = gridspec.GridSpec(2, 2, figure=fig, left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.1, hspace=0.2)
                ax1 = fig.add_subplot(gs[0, 0])
                ax1.hist(dtwpi[dtwpi < N], bins=100)
                a = (dtwpi < N).sum()
                b = len(dtwpi)
                ax1.set_title('count {}(<{})/{}={:.2f}'.format(a, N, b, a/b))
                ax1.set_xlabel(r'W-dist/ns')
                ax2 = fig.add_subplot(gs[0, 1])
                ax2.hist(dtppi, bins=100)
                ax2.set_xlabel(r'P-dist')
                ax3 = fig.add_subplot(gs[1, 0])
                ax3.hist(pediff, bins=100)
                ax3.set_xlabel(r'Adjust PE diff')
                ax4 = fig.add_subplot(gs[1, 1])
                bins = np.linspace(-50, 50, 100)
                ax4.hist(rss_recon - rss_truth, bins=bins, density=1)
                ax4.set_xlabel(r'$\mathrm{RSS}_{recon} - \mathrm{RSS}_{truth}/\mathrm{mV}^{2}$, within (-100, 100)')
                fig.suptitle(args.ipt.split('/')[-1] + ' PEnum={:.0f}'.format(i+1))
                pdf.savefig(fig)
                plt.close()
            else:
                wdist_stats[i, :] = np.nan
                pdist_stats[i, :] = np.nan
            print('\rDrawing Process:|{}>{}|{:6.2f}%'.format(((20*i)//l)*'-', (19-(20*i)//l)*' ', 100*(i+1)/l), end='' if i != l - 1 else '\n')

        plt.rcParams['figure.figsize'] = (12, 6)
        fig = plt.figure()
        gs = gridspec.GridSpec(2, 2, figure=fig, left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.2, hspace=0.2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(dt['wdist'][dt['wdist']<N], bins=100, density=1)
        a = (dt['wdist'] < N).sum()
        b = len(dt['wdist'])
        ax1.set_title('count {}(Wd<{})/{}={:.2f}'.format(a, N, b, a/b))
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(dt['pdist'], bins=100, density=1)
        ax3 = fig.add_subplot(gs[:, 1])
        dtwdistvali = np.logical_and(dt['wdist']<N, np.logical_and(dt['wdist']!=0, dt['pdist']!=0))
        h2 = ax3.hist2d(dt['wdist'][dtwdistvali], dt['pdist'][dtwdistvali], bins=(200, 200), cmap=mycmp)
        fig.colorbar(h2[3], ax=ax3, aspect=50)
        ax3.set_xlabel(r'W-dist/ns')
        ax3.set_ylabel(r'P-dist')
        ax3.set_title('W&P-dist histogram, Wd<{}, flawed'.format(N))
        fig.suptitle(args.ipt.split('/')[-1] + ' Dist stats, method = ' + str(method))
        plt.close()
        pdf.savefig(fig)

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
        ax1.set_xlabel(r'PEnum')
        ax1.set_ylabel(r'W-dist')
        ax1.set_title(r'W-dist vs PEnum stats')
        ax1.legend()
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(pdist_stats[:, 0], c='C0', label='P median')
        ax2.plot(pdist_stats[:, 0] + pdist_stats[:, 1], c='C1', label='P median + mad')
        ax2.plot(pdist_stats[:, 0] - pdist_stats[:, 1], c='C1', label='P median - mad')
        ax2.plot(pdist_stats[:, 2], c='C2', label='P mean')
        ax2.plot(pdist_stats[:, 2] + pdist_stats[:, 3], c='C3', label='P mean + std')
        ax2.plot(pdist_stats[:, 2] - pdist_stats[:, 3], c='C3', label='P mean - std')
        ax2.set_xlabel(r'PEnum')
        ax2.set_ylabel(r'P-dist')
        ax2.set_title(r'P-dist vs PEnum stats')
        ax2.legend()
        fig.suptitle(args.ipt.split('/')[-1] + ' Dist stats, method = ' + str(method))
        plt.close()
        pdf.savefig(fig)

        pdf.close()
