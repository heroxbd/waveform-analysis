# -*- COding: utf-8 -*-


import numpy as np
import csv
import h5py
import argparse

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', help='output')
psr.add_argument('ipt', help='input')
args = psr.parse_args()

if __name__ == '__main__':
    with h5py.File(args.ipt, 'r', libver='latest', swmr=True) as distfile:
        dt = distfile['Record']
        l = len(dt)
        totTime = dt.attrs['totalTime']
        totLen = dt.attrs['totalLength']
        wd = dt['wdist'].mean()
        pd = dt['pdist'].mean()
        penum, c = np.unique(dt['PEnum'], return_counts=True)
        pe_c = np.zeros((len(penum), 2)).astype(np.uint16)
        pe_dist = np.zeros((len(penum), 2))
        pe_c[:, 0] = penum
        pe_c[:, 1] = c
        for i in range(len(penum)):
            pe_dist[i, 0] = np.mean(dt['wdist'][dt['PEnum'] == penum[i]])
            pe_dist[i, 1] = np.mean(dt['pdist'][dt['PEnum'] == penum[i]])
    with open(args.opt, 'w+') as csvf:
        csvwr = csv.writer(csvf)
        csvwr.writerow([args.ipt, str(totTime), str(totLen), str(wd), str(pd)])
        str_pe_c = pe_c.astype(np.str)
        str_pe_dist = pe_dist.astype(np.str)
        for i in range(len(str_pe_dist)):
            csvwr.writerow([str_pe_c[i,0], str_pe_c[i,1], str_pe_dist[i,0], str_pe_dist[i,1]])
