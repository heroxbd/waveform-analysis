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
    with open(args.opt, 'w+') as csvf:
        csvwr = csv.writer(csvf)
        csvwr.writerow([args.ipt, str(totTime), str(totLen), str(wd), str(pd)])
