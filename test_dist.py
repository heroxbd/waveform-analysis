# -*- coding: utf-8 -*-

import sys
import numpy as np
import csv
import h5py
import scipy.stats
import itertools as it
import argparse
import wf_func as wff

psr = argparse.ArgumentParser()
psr.add_argument('--ref', dest='ref', help='reference file', nargs='+')
psr.add_argument('ipt', help="input file")
psr.add_argument('-o', dest='opt', help='output file')
psr.add_argument('-p', dest='print', action='store_false', help='print bool', default=True)
args = psr.parse_args()

if args.print:
    sys.stdout = None

def wpdistance(df_ans, df_sub, df_wav):
    Chnum = len(np.unique(df_ans['ChannelID']))
    e_ans = df_ans['EventID']*Chnum + df_ans['ChannelID']
    e_ans, i_ans = np.unique(e_ans, return_index=True)
    gl = len(e_ans)

    opdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint32), ('PEnum', np.uint16), ('wdist', np.float32), ('pdist', np.float32), ('RSS_recon', np.float32), ('RSS_truth', np.float32), ('PEdiff', np.float32)])
    dt = np.zeros(gl, dtype=opdt)

    leng = len(df_wav[0]['Waveform'])
    e_wav = df_wav['EventID']*Chnum + df_wav['ChannelID']
    e_wav, i_wav = np.unique(e_wav, return_index=True)

    e_sub = df_sub['EventID']*Chnum + df_sub['ChannelID']
    e_sub, i_sub = np.unique(e_sub, return_index=True)
    i_sub = np.append(i_sub, len(df_sub))

    p = 0
    ejd = e_sub[p]

    spe_pre = wff.read_model(args.ref[1])
    for eid, c, i0, i in zip(e_ans, range(gl), np.nditer(i_ans), it.chain(np.nditer(i_ans[1:]), [len(df_ans)])):
        cid = eid % Chnum
        while ejd < eid:
            p += 1
            ejd = e_sub[p]
        assert ejd == eid, 'Answer must include Event {} Channel {}.'.format(eid//Chnum, cid)

        j0 = i_sub[p]; j = i_sub[p+1]
        k0 = i_wav[p]

        wave = wff.deduct_base(spe_pre[cid]['epulse'] * df_wav[k0]['Waveform'], spe_pre[cid]['m_l'], spe_pre[cid]['thres'], 20, 'detail')

        pet_tru = df_ans[i0:i]['PETime']
        pet0, pwe0 = np.unique(pet_tru, return_counts=True)
        pf0 = np.zeros(leng); pf0[pet0] = pwe0
        wave0 = np.convolve(spe_pre[cid]['spe'], pf0, 'full')[:leng]

        wl = df_sub[j0:j]['Weight']
        pet_ans = df_sub[j0:j]['PETime']
        wave1 = wff.showwave(pet_ans, wl, spe_pre[cid]['spe'], leng)

        dt['wdist'][c] = scipy.stats.wasserstein_distance(pet_tru, pet_ans, v_weights=wl)
        Q = i-i0; q = np.sum(wl)
        dt['PEnum'][c] = Q
        dt['pdist'][c] = np.abs(Q - q) * scipy.stats.poisson.pmf(Q, Q)
        dt['EventID'][c] = eid//Chnum
        dt['ChannelID'][c] = cid
        dt['RSS_truth'][c] = np.power(wave0 - wave, 2).sum()
        dt['RSS_recon'][c] = np.power(wave1 - wave, 2).sum()
        if 'PEdiff' in df_sub.dtype.names:
            dt['PEdiff'][c] = df_sub[j0]['PEdiff']
        print('\rGrading Process:|{}>{}|{:6.2f}%'.format(((20 * c)//gl)*'-', (19-(20*c)//gl)*' ', 100 * ((c+1)/gl)), end = '' if c != gl-1 else '\n')
    return dt

def main(fref, fipt, fopt):
    with h5py.File(fref, 'r', libver='latest', swmr=True) as ref, h5py.File(fipt, 'r', libver='latest', swmr=True) as ipt:
        df_ans = ref['GroundTruth']
        df_wav = ref['Waveform']
        df_sub = ipt['Answer']
        method = df_sub.attrs['Method']
        dt = wpdistance(df_ans, df_sub, df_wav)
    with h5py.File(fopt, 'w') as h5f:
        dset = h5f.create_dataset('Record', data=dt, compression='gzip')
        dset.attrs['Method'] = method

if __name__ == '__main__':
    main(args.ref[0], args.ipt, args.opt)
