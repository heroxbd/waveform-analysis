# -*- coding: utf-8 -*-

import sys
import h5py
import numpy as np
import argparse
import wf_func as wff

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', help='output file')
psr.add_argument('ipt', help='input file')
psr.add_argument('-p', dest='pri', action='store_false', help='print bool', default=True)
args = psr.parse_args()

if args.pri:
    sys.stdout = None

def main(fopt, fipt):
    opdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint32), ('PETime', np.uint16), ('Weight', np.uint8), ('PEdiff', np.float32)])
    with h5py.File(fipt, 'r', libver='latest', swmr=True) as ipt:
        method = ipt['Answer'].attrs['Method']
        N = len(ipt['Answer'])
        dt = np.zeros(N, dtype=opdt)
        Eid = ipt['Answer']['EventID']
        Cid = ipt['Answer']['ChannelID']
        Pet = ipt['Answer']['PETime']
        Wgt = ipt['Answer']['Weight']
        Chnum = len(np.unique(Cid))
        Aid = Eid*Chnum + Cid
        e_ans, i_ans, c_ans = np.unique(Aid, return_index=True, return_counts=True)
        start = 0
        end = 0
        l = len(e_ans)
        for i in range(l):
            pet = Pet[i_ans[i]:i_ans[i]+c_ans[i]]
            pwe = Wgt[i_ans[i]:i_ans[i]+c_ans[i]]
            pet_a, pwe_a = wff.xpp_convol(pet, pwe)
            pe_var = np.sum(pwe_a) - np.sum(pwe)

            lenpf = len(pwe_a)
            end = start + lenpf
            dt['PETime'][start:end] = pet_a
            dt['Weight'][start:end] = pwe_a
            dt['EventID'][start:end] = e_ans[i]//Chnum
            dt['ChannelID'][start:end] = e_ans[i]%Chnum
            dt['PEdiff'][start] = pe_var
            start = end
            print('\rAdjusting result:|{}>{}|{:6.2f}%'.format(((20*i)//l)*'-', (19 - (20*i)//l)*' ', 100 * ((i+1) / l)), end=''if i != l-1 else '\n')
    dt = dt[dt['Weight'] > 0]
    with h5py.File(fopt, 'w') as final:
        dset = final.create_dataset('Answer', data=dt, compression='gzip')
        dset.attrs['Method'] = method
        print('The output file path is {}'.format(fopt), end=' ', flush=True)

if __name__ == '__main__':
    main(args.opt, args.ipt)
