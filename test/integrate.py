# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import h5py
import argparse

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', help='output file')
psr.add_argument('ipt', nargs='+', help='input file')
psr.add_argument('--num', type=int, help='fragment number')
psr.add_argument('-p', dest='print', action='store_false', help='print bool', default=True)
args = psr.parse_args()

if args.print:
    sys.stdout = None

def main(unad_path, fopt, upnum):
    opdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint32), ('PETime', np.uint16), ('Weight', np.float16)])
    N = totlen(unad_path, upnum)
    dt = np.zeros(N, dtype=opdt)
    num = 0
    for i in range(len(unad_path)):
        with h5py.File(unad_path[i], 'r', libver='latest', swmr=True) as up:
            dt[num:num+len(up['Answer'])] = up['Answer']
            num += len(up['Answer'])
    dt = dt[:num]
    with h5py.File(fopt, 'w') as adj:
        adj.create_dataset('Answer', data=dt, compression='gzip')
        print('The output file path is {}'.format(fopt), end=' ', flush=True)

def totlen(unad_path, num):
    totl = 0
    n = min(len(unad_path), 3)
    for i in range(n):
        with h5py.File(unad_path[i], 'r', libver='latest', swmr=True) as up:
            totl = totl + len(up['Answer']['PETime']) * ((num+1)*3//2)
    totl = totl // n
    return totl

if __name__ == '__main__':
    main(args.ipt, args.opt, args.num)
