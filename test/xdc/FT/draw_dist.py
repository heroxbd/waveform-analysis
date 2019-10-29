import h5py
import argparse

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', help='output')
psr.add_argument('ipt', help='input')
args = psr.parse_args()

import csv
import numpy as np
import matplotlib.pyplot as plt 

plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 18
plt.rcParams['lines.markersize'] = np.sqrt(10)
plt.rcParams['lines.linewidth'] = 1.0

if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = (16, 9)
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 18
    plt.rcParams['lines.markersize'] = np.sqrt(10)
    plt.rcParams['lines.color'] = 'g'
    plt.rcParams['lines.linewidth'] = 1.0
    distfile = h5py.File(args.ipt)
    dt = distfile['Record']
    l = len(dt)
    plt.subplot(221)
    plt.hist(dt['wdist'], bins=100, density=1)
    plt.title(r'W-dist histogram')
    plt.subplot(223)
    plt.hist(dt['pdist'], bins=100, density=1)
    plt.title(r'P-dist histogram')
    plt.subplot(122)
    plt.hist2d(dt['wdist'], dt['pdist'], bins=(100, 100), density=1)
    plt.xlabel('W-dist')
    plt.ylabel('P-dist')
    plt.title(r'W&P-dist histogram')
    plt.savefig(args.opt)
    plt.close()
    distfile.close()
