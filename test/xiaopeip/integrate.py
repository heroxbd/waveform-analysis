# -*- coding: utf-8 -*-

import os
import numpy as np
import h5py
import argparse

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', help='output')
psr.add_argument('ipt', nargs='+', help='input')
psr.add_argument('--num', type=int)
args = psr.parse_args()

def main(unad_path, fopt, upnum):
    opdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint8), ('PETime', np.uint16), ('Weight', np.float16)])
    N = totlen(unad_path[0], upnum)
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
    with h5py.File(unad_path, 'r', libver='latest', swmr=True) as up:
        totl = len(up['Answer']['PETime']) * (num+2) * 2 // 3
    return totl

if __name__ == '__main__':
    main(args.ipt, args.opt, args.num)
