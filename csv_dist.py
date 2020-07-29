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
psr.add_argument('--mod', type=str, help='mode of weight or charge', choices=['PEnum', 'Charge'])
psr.add_argument('-p', dest='pri', action='store_false', help='print bool', default=True)
args = psr.parse_args()
mode = args.mod
if mode == 'PEnum':
    extradist = 'pdist'
    pecount = 'TotalPEnum'
elif mode == 'Charge':
    extradist = 'chargediff'
    pecount = 'TotalPEpos'
if args.pri:
    sys.stdout = None

with h5py.File(args.ipt, 'r', libver='latest', swmr=True) as distfile:
    dt = distfile['Record'][:]
    l = len(dt)
    wd = dt['wdist'].mean()
    pd = dt[extradist].mean()
    penum, c = np.unique(dt[pecount], return_counts=True)
    pe_c = np.zeros((len(penum), 2)).astype(np.uint32)
    pe_dist = np.zeros((len(penum), 2))
    pe_c[:, 0] = penum
    pe_c[:, 1] = c
    for i in tqdm(range(len(penum)), disable=args.pri):
        pe_dist[i, 0] = np.mean(dt['wdist'][dt[pecount] == penum[i]])
        pe_dist[i, 1] = np.mean(dt[extradist][dt[pecount] == penum[i]])
with open(args.opt, 'w+') as csvf:
    csvwr = csv.writer(csvf)
    csvwr.writerow([args.ipt, str(wd), str(pd)])
    str_pe_c = pe_c.astype(np.str)
    str_pe_dist = pe_dist.astype(np.str)
    for i in range(len(str_pe_dist)):
        csvwr.writerow([str_pe_c[i,0], str_pe_c[i,1], str_pe_dist[i,0], str_pe_dist[i,1]])
