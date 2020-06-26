# -*- coding: utf-8 -*-

import numpy as np
import h5py
import argparse

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', help='output file')
psr.add_argument('ipt', help='input file')
psr.add_argument('-a', type=int, help='begining of fragment')
psr.add_argument('-b', type=int, help='ending of fragment')
args = psr.parse_args()

fopt = args.opt
fipt = args.ipt
a = args.a
b = args.b

with h5py.File(fipt, 'r', libver='latest', swmr=True) as ipt:
    Tr = ipt['GroundTruth']
    Wf = ipt['Waveform']
    Chnum = len(np.unique(Wf['ChannelID']))
    Wf_num = Wf['EventID'] * Chnum + Wf['ChannelID']
    Wf_num = Wf_num - Wf_num[0]
    Tr_num = Tr['EventID'] * Chnum + Tr['ChannelID']
    Tr_num = Tr_num - Tr_num[0]
    if b > len(Wf_num):
        print('b exceeded Waveform_num which is {}'.format(len(Wf_num)))
    if a < 0 and b > 0:
        Nb = Wf_num[b]
        wf = Wf[Wf_num < Nb]
        tr = Tr[Tr_num < Nb]
    elif b <= 0 and a >= 0:
        Na = Wf_num[a]
        wf = Wf[Wf_num >= Na]
        tr = Tr[Tr_num >= Na]
    elif a < 0 and b <= 0:
        wf = Wf[:]
        tr = Tr[:]
    else:
        Na = Wf_num[a]
        Nb = Wf_num[b]
        wf = Wf[np.logical_and(Wf_num >= Na, Wf_num < Nb)]
        tr = Tr[np.logical_and(Tr_num >= Na, Tr_num < Nb)]
with h5py.File(fopt, 'w') as opt:
    opt.create_dataset('GroundTruth', data=tr, compression='gzip')
    opt.create_dataset('Waveform', data=wf, compression='gzip')
