# -*- coding: utf-8 -*-

import h5py
import argparse

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', help='output')
psr.add_argument('ipt', help='input')
psr.add_argument('--mode', type=int, dest='mode', help='mode of draw')
args = psr.parse_args()

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 4
plt.rcParams['lines.markersize'] = np.sqrt(10)
plt.rcParams['lines.linewidth'] = 1.0

def my_cmap():
    plasma = cm.get_cmap('plasma', 256)
    newcolors = plasma(np.linspace(0, 1, 256))
    white = np.array([255/256, 255/256, 255/256, 1])
    newcolors[:1, :] = white
    newcmp = ListedColormap(newcolors)
    return newcmp

def draw_pe_w(fig, gs, i_d, dt_wdist, dt_penum, penum):
    if i_d in penum:
        ax = fig.add_subplot(gs[(i_d-1)//10, (i_d-1)%10])
        ax.hist(dt_wdist[dt_penum == i_d], bins=20)
        ax.set_title('PEnum={}, '.format(i_d) + r'$\bar{Wdist}$' + '={}'.format(np.mean(dt_wdist[dt_penum == i_d])))
    return

def draw_pe_p(fig, gs, i_d, dt_pdist, dt_penum, penum):
    if i_d in penum:
        ax = fig.add_subplot(gs[(i_d-1)//10, (i_d-1)%10])
        ax.hist(dt_pdist[dt_penum == i_d], bins=20)
        ax.set_title('PEnum={}, '.format(i_d) + r'$\bar{Pdist}$' + '={}'.format(np.mean(dt_pdist[dt_penum == i_d])))
    return

if __name__ == '__main__':
    mycmp = my_cmap()
    with h5py.File(args.ipt, 'r', libver='latest', swmr=True) as distfile:
            dt = distfile['Record']
            if args.mode == 0:
                plt.rcParams['figure.figsize'] = (16, 12)
                fig = plt.figure()
                gs = gridspec.GridSpec(2, 2, figure=fig)
                ax1 = fig.add_subplot(gs[0, 0])
                ax1.hist(dt['wdist'], bins=100, density=1)
                ax1.set_title(r'W-dist histogram')
                ax2 = fig.add_subplot(gs[1, 0])
                ax2.hist(dt['pdist'], bins=100, density=1)
                ax2.set_title(r'P-dist histogram')
                ax3 = fig.add_subplot(gs[:, 1])
                h2 = ax3.hist2d(dt['wdist'], dt['pdist'], bins=(100, 100), cmap=mycmp)
                fig.colorbar(h2[3], ax=ax3, aspect=50)
                ax3.set_xlabel(r'W-dist')
                ax3.set_ylabel(r'P-dist')
                ax3.set_title(r'W&P-dist histogram')
                fig.suptitle(args.ipt.split('/')[-1])
                fig.savefig(args.opt)
            elif args.mode == 1:
                plt.rcParams['figure.figsize'] = (20, 20)
                penum = np.unique(dt['PEnum'])
                penum = np.sort(penum)
                fig = plt.figure()
                gs = gridspec.GridSpec(10, 10, figure=fig)
                for i in np.arange(1, min(100, penum.max()) + 1):
                    draw_pe_w(fig, gs, i, dt['wdist'], dt['PEnum'], penum)
                fig.suptitle(args.ipt.split('/')[-1] + ' ' + r'W-dist vs PEnum')
                fig.savefig(args.opt)
            elif args.mode == 2:
                plt.rcParams['figure.figsize'] = (20, 20)
                penum = np.unique(dt['PEnum'])
                penum = np.sort(penum)
                fig = plt.figure()
                gs = gridspec.GridSpec(10, 10, figure=fig)
                for i in np.arange(1, min(100, penum.max()) + 1):
                    draw_pe_p(fig, gs, i, dt['pdist'], dt['PEnum'], penum)
                fig.suptitle(args.ipt.split('/')[-1] + ' ' + r'P-dist vs PEnum')
                fig.savefig(args.opt)
