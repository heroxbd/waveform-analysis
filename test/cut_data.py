# -*- coding: utf-8 -*-

import numpy as np
import h5py
import argparse

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', help='output file')
psr.add_argument('ipt', help='input file')
psr.add_argument('-a', type=int, help='begining of gragment')
psr.add_argument('-b', type=int, help='ending of gragment')
args = psr.parse_args()

def main(fopt, fipt, a, b):
    with h5py.File(fipt, 'r', libver='latest', swmr=True) as ipt:
        Tr = ipt['GroundTruth']
        Wf = ipt['Waveform']
        if a < 0 and b > 0:
            wf = Wf[Wf['EventID'] < b]
            tr = Tr[Tr['EventID'] < b]
        elif b <= 0 and a >= 0:
            wf = Wf[Wf['EventID'] >= a]
            tr = Tr[Tr['EventID'] >= a]
        elif a < 0 and b <= 0:
            wf = Wf[:]
            tr = Tr[:]
        else:
            wf = Wf[np.logical_and(Wf['EventID'] >= a, Wf['EventID'] < b)]
            tr = Tr[np.logical_and(Tr['EventID'] >= a, Tr['EventID'] < b)]
    with h5py.File(fopt, 'w') as opt:
        dset = opt.create_dataset('GroundTruth', data=tr, compression='gzip')
        dset = opt.create_dataset('Waveform', data=wf, compression='gzip')

if __name__ == '__main__':
    main(args.opt, args.ipt, args.a, args.b)
