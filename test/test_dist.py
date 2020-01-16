# -*- coding: utf-8 -*-

import numpy as np
import csv
import h5py
import scipy.stats
import itertools as it
import argparse

psr = argparse.ArgumentParser()
psr.add_argument('--ref', dest='ref', help='reference file')
psr.add_argument('ipt', help="input file")
psr.add_argument('-o', dest='opt', help='output file')
psr.add_argument('-p', dest='print', action='store_false', help='print bool', default=True)
args = psr.parse_args()

if args.print:
    sys.stdout = None

def wpdistance(df_ans, df_sub):
    Chnum = np.max(df_ans['ChannelID'])
    e_ans = df_ans['EventID']*Chnum + df_ans['ChannelID']
    e_ans, i_ans = np.unique(e_ans, return_index=True)
    gl = len(e_ans)

    opdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint8), ('PEnum', np.uint16), ('wdist', np.float32), ('pdist', np.float32)])
    dt = np.zeros(gl, dtype=opdt)

    e_sub = df_sub['EventID']*Chnum + df_sub['ChannelID']
    e_sub, i_sub = np.unique(e_sub, return_index=True)
    i_sub = np.append(i_sub, len(df_sub))

    p = 0
    ejd = e_sub[p]

    for eid, c, i0, i in zip(e_ans, range(gl), np.nditer(i_ans), it.chain(np.nditer(i_ans[1:]), [len(df_ans)])):
        while ejd < eid:
            p += 1
            ejd = e_sub[p]
        assert ejd == eid, 'Answer must include Event {} Channel {}.'.format(eid//Chnum, eid % Chnum)

        j0 = i_sub[p]; j = i_sub[p+1]

        wl = df_sub[j0:j]['Weight']
        dt['wdist'][c] = scipy.stats.wasserstein_distance(df_ans[i0:i]['PETime'], df_sub[j0:j]['PETime'], v_weights=wl)
        dt['wdist'][c] = scipy.stats.wasserstein_distance(df_ans[i0:i]['PETime'], df_sub[j0:j]['PETime'], v_weights=wl)
        Q = i-i0; q = np.sum(wl)
        dt['PEnum'][c] = Q
        dt['pdist'][c] = np.abs(Q - q) * scipy.stats.poisson.pmf(Q, Q)
        dt['EventID'][c] = eid//Chnum
        dt['ChannelID'][c] = eid % Chnum
        print('\rGrading Process:|{}>{}|{:6.2f}%'.format(((20 * c)//gl)*'-', (19-(20*c)//gl)*' ', 100 * ((c+1)/gl)), end = '' if c != gl-1 else '\n')
    return dt

def main(ref, ipt, opt):
    with h5py.File(ref, 'r', libver='latest', swmr=True) as ref, h5py.File(ipt, 'r', libver='latest', swmr=True) as ipt:
        df_ans = ref['GroundTruth']
        df_sub = ipt['Answer']
        dt = wpdistance(df_ans, df_sub)
    with h5py.File(opt, 'w') as h5f:
        dset = h5f.create_dataset('Record', data=dt, compression='gzip')

if __name__ == '__main__':
    main(args.ref, args.ipt, args.opt)
