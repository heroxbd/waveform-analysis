# -*- coding: utf-8 -*-

import sys
import h5py
import numpy as np
import argparse
from shutil import copyfile
from multiprocessing import Pool
import wf_func as wff

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', help='output file')
psr.add_argument('ipt', help='input file', nargs='+')
psr.add_argument('--mod', type=str, help='mode of pe or charge')
psr.add_argument('--ref', type=str, help='reference file')
psr.add_argument('-N', dest='Ncpu', type=int, help='cpu number', default=50)
psr.add_argument('-p', dest='pri', action='store_false', help='print bool', default=True)
args = psr.parse_args()

fopt = args.opt
fipt = args.ipt
reference = args.ref
mode = args.mod

Ncpu = args.Ncpu
Thres = 0.2

if args.pri:
    sys.stdout = None

def select(a, b):
    spe_pre = wff.read_model(reference)
    with h5py.File(fipt[0], 'r', libver='latest', swmr=True) as ipt:
        N = len(ipt['Answer'])
        Eid = ipt['Answer']['EventID']
        Cid = ipt['Answer']['ChannelID']
        Pet = ipt['Answer']['PETime']
        Wgt = ipt['Answer']['Weight']
        Chnum = len(np.unique(Cid))
        Aid = Eid * Chnum + Cid
        e_ans, i_ans, c_ans = np.unique(Aid, return_index=True, return_counts=True)
        start = 0
        end = 0
        fi = h5py.File(fipt[1], 'r', libver='latest', swmr=True)
        Wf = fi['Waveform']
        dt = np.zeros(np.sum(c_ans[a:b]), dtype=opdt)
        for i in range(a, b):
            cid = e_ans[i]%Chnum
            pet = Pet[i_ans[i]:i_ans[i]+c_ans[i]]
            pwe = Wgt[i_ans[i]:i_ans[i]+c_ans[i]]
            wave = wff.deduct_base(spe_pre[cid]['epulse'] * Wf[i]['Waveform'], spe_pre[cid]['m_l'], spe_pre[cid]['thres'], 20, 'detail')
            pet_a, pwe_a = wff.hybird_select(pet, pwe, wave, spe_pre[cid]['spe'], Thres)
            pe_var = np.sum(pwe_a) - np.sum(pwe)
            lenpf = len(pwe_a)
            end = start + lenpf
            dt['PETime'][start:end] = pet_a
            dt['Weight'][start:end] = pwe_a
            dt['EventID'][start:end] = e_ans[i]//Chnum
            dt['ChannelID'][start:end] = cid
            dt['PEdiff'][start] = pe_var
            start = end
        fi.close()
    dt = dt[dt['Weight'] > 0]
    return dt

if mode == 'Weight':
    opdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint32), ('PETime', np.uint16), ('Weight', np.uint8), ('PEdiff', np.float32)])
    with h5py.File(fipt[0], 'r', libver='latest', swmr=True) as fi:
        method = fi['Answer'].attrs['Method']
    with h5py.File(fipt[1], 'r', libver='latest', swmr=True) as fi:
        l = len(fi['Waveform'])
    chunk = l // Ncpu + 1
    slices = np.vstack((np.arange(0, l, chunk), np.append(np.arange(chunk, l, chunk), l))).T.astype(np.int).tolist()
    with Pool(Ncpu) as pool:
        select_result = pool.starmap(select, slices)
    result = np.hstack(select_result)
    with h5py.File(fopt, 'w') as final:
        dset = final.create_dataset('Answer', data=result, compression='gzip')
        dset.attrs['Method'] = method
elif mode == 'Charge':
    copyfile(fipt[0], fopt)
print('The output file path is {}'.format(fopt))
