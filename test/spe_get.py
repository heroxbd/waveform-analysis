# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse

psr = argparse.ArgumentParser()
psr.add_argument('ipt', nargs='+', help='input file')
psr.add_argument('-o', dest='opt', help='output file')
psr.add_argument('--num', dest='spenum', type=int, help='num of speWf')
psr.add_argument('--len', dest='spelen', type=int, help='length of speWf')
psr.add_argument('-p', dest='print', action='store_false', help='print bool', default=True)
args = psr.parse_args()

N = args.spenum
L = args.spelen
if args.print:
    sys.stdout = None

def generate_standard(h5_path, single_pe_path):
    npdt = np.dtype([('TrainSet', np.uint8), ('EventID', np.uint32), ('ChannelID', np.uint32), ('speWf', np.uint16, L)])  # set datatype
    dt = np.zeros(N, dtype=npdt)
    num = 0

    for i in range(len(h5_path)):
        with h5py.File(h5_path[i], 'r', libver='latest', swmr=True) as ztrfile:
            ptev = ztrfile['GroundTruth']['EventID']
            ptch = ztrfile['GroundTruth']['ChannelID']
            Pt = ztrfile['GroundTruth']['PETime']
            wfev = ztrfile['Waveform']['EventID']
            wfch = ztrfile['Waveform']['ChannelID']
            Wf = ztrfile['Waveform']['Waveform']
            ni = 0
            Length_pe = len(Wf[0])
            for j in range(len(Wf)):
                wf = Wf[j]
                pt = (np.around(np.sort(Pt[np.logical_and(ptev == wfev[j], ptch == wfch[j])])) + 0.1).astype(int)
                pt = pt[pt >= 0]
                if len(pt) == 1 and pt[0] < Length_pe - L:
                    ps = pt
                else:
                    dpta = np.diff(pt, prepend=pt[0])
                    dptb = np.diff(pt, append=pt[-1])
                    ps = pt[np.logical_and(dpta > L, dptb > L)]#long distance to other spe in both forepart & backpart
                    if dpta[-1] > L and pt[-1] < Length_pe - L:
                        ps = np.insert(ps, 0, pt[-1])
                    if dptb[0] > L and pt[0] < Length_pe - L:
                        ps = np.insert(ps, 0, pt[0])
                for k in range(len(ps)):
                    dt[num]['TrainSet'] = i
                    dt[num]['EventID'] = wfev[j]
                    dt[num]['ChannelID'] = wfch[j]
                    dt[num]['speWf'] = wf[ps[k]:ps[k]+L]
                    print('\rSingle PE Generating:|{}>{}|{:6.2f}%'.format(((20*num)//N)*'-', (19 - (20*num)//N)*' ', 100 * ((num+1) / N)), end=''if num != N-1 else '\n')
                    num += 1
                    if num >= N:
                        break
                if num >= N:
                    break
            if num >= N:
                break
    dt = dt[:num] # cut empty dt part
    print('{} speWf generated'.format(len(dt)))
    with h5py.File(single_pe_path, 'w') as spp:
        spp.create_dataset('SinglePE', data=dt, compression='gzip') # save the spe events

def main(h5_path, single_pe_path):
    if not os.path.exists(single_pe_path):
        generate_standard(h5_path, single_pe_path) # generate response model

if __name__ == '__main__':
    main(args.ipt, args.opt)
