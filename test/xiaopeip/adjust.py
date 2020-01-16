# -*- coding: utf-8 -*-

import h5py
import numpy as np
import argparse

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', help='output file')
psr.add_argument('ipt', help='input file')
psr.add_argument('-p', dest='print', action='store_false', help='print bool', default=True)
args = psr.parse_args()

if args.print:
    sys.stdout = None

def main(fopt, fipt):
    opdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint8), ('PETime', np.uint16), ('Weight', np.uint8)])
    with h5py.File(fipt, 'r', libver='latest', swmr=True) as ipt:
        N = len(ipt['Answer'])
        dt = np.zeros(N, dtype=opdt)
        Eid = ipt['Answer']['EventID']
        Cid = ipt['Answer']['ChannelID']
        Pet = ipt['Answer']['PETime']
        Wgt = ipt['Answer']['Weight']
        Chnum = np.max(Cid)
        Aid = Eid*Chnum + Cid
        e_ans, i_ans, c_ans = np.unique(Aid, return_index=True, return_counts=True)
        start = 0
        end = 0
        l = len(e_ans)
        for i in range(l):
            pet = Pet[i_ans[i]:i_ans[i]+c_ans[i]]
            wgt = Wgt[i_ans[i]:i_ans[i]+c_ans[i]][np.argsort(pet)]
            pet = np.sort(pet)
            wgt_b = np.floor(wgt)
            resi = wgt - wgt_b
            t = np.convolve(resi, [0.9, 1.7, 0.9], 'full')
            t = t[1:-1]
            ta = np.diff(t, prepend=t[0])
            tb = np.diff(t, append=t[-1])
            wgt_b[np.logical_and(np.logical_and(ta > 0, tb < 0), resi > 0.5)] += 1
            if len(wgt_b[wgt_b > 0]) == 0:
                wgt = 1
                pet = np.max(pet)
                lenpf = 1
            else:
                wgt = wgt_b[wgt_b > 0]
                pet = pet[wgt_b > 0]
                lenpf = len(wgt)
            end = start + lenpf
            dt['PETime'][start:end] = pet
            dt['Weight'][start:end] = wgt
            dt['EventID'][start:end] = e_ans[i]//Chnum
            dt['ChannelID'][start:end] = e_ans[i]%Chnum
            start = end
            print('\rAdjusting result:|{}>{}|{:6.2f}%'.format(((20*i)//l)*'-', (19 - (20*i)//l)*' ', 100 * ((i+1) / l)), end=''if i != l-1 else '\n')
    dt = dt[dt['Weight'] > 0]
    with h5py.File(fopt, 'w') as final:
        final.create_dataset('Answer', data=dt, compression='gzip')
        print('The output file path is {}'.format(fopt), end=' ', flush=True)

if __name__ == '__main__':
    main(args.opt, args.ipt)
