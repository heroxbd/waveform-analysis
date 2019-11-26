# -*- coding: utf-8 -*-

import h5py
import argparse

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', help='output')
psr.add_argument('ipt', help='input')
args = psr.parse_args()

import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 18
plt.rcParams['lines.markersize'] = np.sqrt(10)
plt.rcParams['lines.linewidth'] = 1.0

def my_cmap():
    plasma = cm.get_cmap('plasma', 256)
    newcolors = plasma(np.linspace(0, 1, 256))
    white = np.array([255/256, 255/256, 255/256, 1])
    newcolors[:1, :] = white
    newcmp = ListedColormap(newcolors)
    return newcmp

if __name__ == '__main__':
    mycmp = my_cmap()
    with h5py.File(args.ipt, 'r', libver='latest', swmr=True) as distfile:
        dt = distfile['Record']
        l = len(dt)
        plt.subplot(221)
        plt.hist(dt['wdist'], bins=100, density=1)
        plt.title(r'W-dist histogram')
        plt.subplot(223)
        plt.hist(dt['pdist'], bins=100, density=1)
        plt.title(r'P-dist histogram')
        plt.subplot(122)
        plt.hist2d(dt['wdist'], dt['pdist'], bins=(100, 100), cmap=mycmp)
        plt.colorbar(aspect=50)
        plt.xlabel('W-dist')
        plt.ylabel('P-dist')
        plt.title(r'W&P-dist histogram')
        plt.suptitle(dt.attrs['spePath'].split('/')[-1] + ' ' + args.ipt.split('/')[-1])
        plt.savefig(args.opt)
        plt.close()
