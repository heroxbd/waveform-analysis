# -*- COding: utf-8 -*-

import sys
import numpy as np
import csv
from tqdm import tqdm
import h5py
import argparse

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', help='output')
psr.add_argument('ipt', help='input')
psr.add_argument('--ref', type=str, nargs='+', help='reference file')
psr.add_argument('-p', dest='pri', action='store_false', help='print bool', default=True)
args = psr.parse_args()
if args.pri:
    sys.stdout = None

with h5py.File(args.ipt, 'r', libver='latest', swmr=True) as distfile:
    dt = distfile['Record'][:]
    l = len(dt)
    wd = dt['wdist'].mean()
    stdwd = dt['wdist'].std()
    pwd1 = np.percentile(dt['wdist'], 5)
    pwd2 = np.percentile(dt['wdist'], 95)
    pd = dt['chargediff'].mean()
    stdpd = dt['chargediff'].std()
    ppd1 = np.percentile(dt['chargediff'], 5)
    ppd2 = np.percentile(dt['chargediff'], 95)
    penum, c = np.unique(dt['TotalPEpos'], return_counts=True)
    pe_c = np.zeros((len(penum), 2)).astype(np.uint32)
    pe_dist = np.zeros((len(penum), 8))
    pe_c[:, 0] = penum
    pe_c[:, 1] = c
    for i in tqdm(range(len(penum)), disable=args.pri):
        pe_dist[i, 0] = np.mean(dt['wdist'][dt['TotalPEpos'] == penum[i]])
        pe_dist[i, 1] = np.std(dt['wdist'][dt['TotalPEpos'] == penum[i]])
        pe_dist[i, 2] = np.percentile(dt['wdist'][dt['TotalPEpos'] == penum[i]], 5)
        pe_dist[i, 3] = np.percentile(dt['wdist'][dt['TotalPEpos'] == penum[i]], 95)
        pe_dist[i, 4] = np.mean(dt['chargediff'][dt['TotalPEpos'] == penum[i]])
        pe_dist[i, 5] = np.std(dt['chargediff'][dt['TotalPEpos'] == penum[i]])
        pe_dist[i, 6] = np.percentile(dt['chargediff'][dt['TotalPEpos'] == penum[i]], 5)
        pe_dist[i, 7] = np.percentile(dt['chargediff'][dt['TotalPEpos'] == penum[i]], 95)
with open(args.opt, 'w+') as csvf:
    csvwr = csv.writer(csvf)
    csvwr.writerow([str(), args.ipt, str(wd), str(stdwd), str(pwd1), str(pwd2), str(pd), str(stdpd), str(ppd1), str(ppd2)])
    str_pe_c = pe_c.astype(np.str)
    str_pe_dist = pe_dist.astype(np.str)
    for i in range(len(str_pe_dist)):
        csvwr.writerow([str_pe_c[i,0], str_pe_c[i,1]] + [str_pe_dist[i,j] for j in range(8)])
