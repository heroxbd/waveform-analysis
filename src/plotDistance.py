# -*- coding: utf-8 -*-

import h5py
import argparse
psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', help='output')
psr.add_argument('ipt', help='input')
args = psr.parse_args()
import pandas as pd
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
plt.rcParams['font.size'] = 16
plt.rcParams['lines.markersize'] = 2
plt.rcParams['lines.linewidth'] = 1.0

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
        pdf = PdfPages(args.opt)

        penum = np.unique(dt['PEnum'])
        penum = np.sort(penum)
        lbegin = penum.min()
        if lbegin == 0:
            lbegin =1
        if (penum.max()-lbegin)<100:
            lend = penum.max() + 1
        else:
            lend = lbegin + 100
        l = lend - lbegin
        wdist_stats = np.zeros((l, 4))
        pdist_stats = np.zeros((l, 4))
        for i in np.arange(l):
            if (i+lbegin) in penum:
                dtwpi = dt['wdist'][dt['PEnum'] == (i+lbegin)]
                dtppi = dt['pdist'][dt['PEnum'] == (i+lbegin)]
                wdist_stats[i, 0] = np.median(dtwpi)
                wdist_stats[i, 1] = np.median(np.absolute(dtwpi - np.median(dtwpi)))
                wdist_stats[i, 2] = np.mean(dtwpi)
                wdist_stats[i, 3] = np.std(dtwpi)
                pdist_stats[i, 0] = np.median(dtppi)
                pdist_stats[i, 1] = np.median(np.absolute(dtppi - np.median(dtppi)))
                pdist_stats[i, 2] = np.mean(dtppi)
                pdist_stats[i, 3] = np.std(dtppi)
                plt.rcParams['figure.figsize'] = (12, 6)
                fig = plt.figure()
                ax1 = fig.add_subplot(121)
                ax1.hist(dtwpi[dtwpi < 10], bins=50)
                a = (dtwpi < 10).sum()
                b = len(dtwpi)
                # ax1.set_title('count {}(<10)/{}={:.2f}'.format(a, b, a/b), fontsize=12)
                ax1.set_title('Wasserstein distance distribution(PE={})'.format(i+lbegin))
                ax1.set_xlabel(r'/ns')
                ax1.set_ylabel('entry number')
                ax2 = fig.add_subplot(122)
                ax2.hist(dtppi, bins=20)
                ax2.set_xlabel(r'PE number')
                ax2.set_ylabel('entry number')
                # ax2.set_title('count={}'.format(len(dtppi)), fontsize=12)
                ax2.set_title('Poisson distance distribution(PE={})'.format(i+lbegin))
                fig.suptitle('PEnum={:.0f}'.format(i+lbegin))
                pdf.savefig(fig)
                plt.close()
            else:
                wdist_stats[i, :] = np.nan
                pdist_stats[i, :] = np.nan
            print('\r Drawing Process:|{}>{}|{:6.2f}%'.format(((20*i)//l)*'-', (19-(20*i)//l)*' ', 100*(i+1)/l), end='' if i != l - 1 else '\n')

        plt.rcParams['figure.figsize'] = (12, 6)
        fig = plt.figure()
        gs = gridspec.GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[:, 0])
        ax1.hist(dt['wdist'][dt['wdist']<100], bins=100, density=1)
        ax1.set_xlabel('/ns')
        a = (dt['wdist'] < 10).sum()
        b = len(dt['wdist'])
        ax1.text(0.6, 1, '$\mu={:.2f}$\n$\sigma={:.2f}$\nmax={:.2f}'.format(np.average(dt['wdist']), np.std(dt['wdist']), np.max(dt['wdist'])), verticalalignment='top', transform=plt.gca().transAxes)

        # ax1.set_title('count {}(Wd<10)/{}={:.2f}'.format(a, b, a/b), fontsize=12)
        ax1.set_title('Wasserstein distance distribution')
        ax2 = fig.add_subplot(gs[:, 1])
        ax2.hist(dt['pdist'], bins=100, density=1)
        ax2.text(0.6, 1, '$\mu={:.2f}$\n$\sigma={:.2f}$\nmax={:.2f}'.format(np.average(dt['pdist']), np.std(dt['pdist']), np.max(dt['pdist'])), verticalalignment='top', transform=plt.gca().transAxes)
        ax2.set_title('Possion distance distribution')
        ax2.set_xlabel('PEnum')
        plt.close()
        pdf.savefig(fig)
        
        fig = plt.figure()
        ax3 = fig.add_subplot(111)
        h2 = ax3.hist2d(dt['wdist'][dt['wdist']<10], dt['pdist'][dt['wdist']<10], bins=(200, 200), cmap=mycmp)
        fig.colorbar(h2[3], ax=ax3, aspect=50)
        ax3.set_xlabel(r'W-dist/ns')
        ax3.set_ylabel(r'P-dist')
        ax3.set_title(r'W&P-dist histogram, Wd<10', fontsize=12)
        plt.close()
        pdf.savefig(fig)

        fig = plt.figure()
        ax4 = fig.add_subplot()
        data = pd.DataFrame({'pe': pd.Series(dt['PEnum']),
                            'distance': pd.Series(dt['wdist'])})
        distanceAverage = data.groupby('pe', as_index=False).mean()
        distanceStd = data.groupby('pe').std()

        x = distanceAverage['pe']
        y = distanceAverage['distance']
        yError = distanceStd['distance']
        plt.errorbar(x, y, yerr=yError, fmt='.')
        plt.title('PEnum-Wdistance')
        pdf.savefig(fig)


        plt.rcParams['figure.figsize'] = (12, 6)
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
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
        ax2 = fig.add_subplot(122)
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
        fig.suptitle(args.ipt.split('/')[-1] + ' Dist stats')
        plt.close()
        pdf.savefig(fig)

        pdf.close()

