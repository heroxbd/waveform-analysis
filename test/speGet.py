#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 19:51:47 2019

@author: xudachengthu

Generate standard response model of single pe's waveform
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', help='output')
psr.add_argument('ipt', nargs='+', help='input')
args = psr.parse_args()

N = 1000
L = 50

def generate_standard(h5_path, single_pe_path):
    npdt = np.dtype([('TrainSet', np.uint8), ('EventID', np.uint32), ('ChannelID', np.uint8), ('speWf', np.uint16, L)])  # set datatype
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
            for j in range(len(Wf)):
                wf = Wf[j]
                pt = (np.around(np.sort(Pt[np.logical_and(ptev == wfev[j], ptch == wfch[j])])) + 0.1).astype(int)
                dpta = np.diff(pt, prepend=pt[0])
                dptb = np.diff(pt, append=pt[-1])
                ps = pt[np.logical_and(dpta > L, dptb > L)]#long distance to other spe in both forepart & backpart
                for k in range(len(ps)):
                    dt['TrainSet'][num] = i
                    dt['EventID'][num] = wfev[j]
                    dt['ChannelID'][num] = wfch[j]
                    dt['speWf'][num] = wf[ps[k]:ps[k]+L]
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
    with h5py.File(single_pe_path, "w") as spp:
        spp.create_dataset('SinglePE', data=dt, compression='gzip') # save the spe events

def speplot(dt):
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 16 # set figure parameters
    
    spemean = np.mean(dt['speWf'], axis = 0) # calculate average fluctuation of waveform
    plt.figure()
    plt.xlim(0,50)
    plt.plot(spemean) # draw the average fluctuation
    plt.xlabel('ns')
    plt.ylabel('mV')
    plt.title("Standard response model")
    #plt.savefig('spemean.png')
    plt.close()
    
    spemin = np.min(dt['speWf'], axis = 1)
    u = np.unique(Pt)
    plt.figure()
    plt.hist(spemin, len(u), density=1, histtype='bar', cumulative=False) # show the dispersion of minimum of spe waveform
    plt.xlabel('mV')
    #plt.savefig('specumu.png')
    plt.close()

def main(h5_path, single_pe_path):
    if not os.path.exists(single_pe_path):
        generate_standard(h5_path, single_pe_path) # generate response model

if __name__ == '__main__':
    main(args.ipt, args.opt)
