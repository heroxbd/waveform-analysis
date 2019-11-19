#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 19:51:47 2019

@author: xudachengthu

Generate standard response model of single pe's waveform
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import time
import argparse

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', help='output')
psr.add_argument('ipt', nargs=10, help='input')
args = psr.parse_args()

N = 10000

def generate_standard(h5_path, single_pe_path):
    npdt = np.dtype([('TrainSet', np.uint8), ('EventID', np.uint32), ('ChannelID', np.uint8), ('Waveform', np.uint16, 1029), ('speWf', np.uint16, 120)]) # set datatype
    
    zf0 = h5py.File(h5_path[0])
    dt = np.zeros(min(len(zf0['Waveform']), N)*10//15, dtype=npdt) # assume ratio of single pe is less than 1/15
    zf0.close()
    num = 0
    l_s = 0
    for i in range(len(h5_path)):

        with h5py.File(h5_path[i], 'r', libver='latest', swmr=True) as ztrfile:
        
            wf = ztrfile['Waveform'] # read waveform only
            answ = pd.read_hdf(h5_path[i], "GroundTruth") # read h5 file answer
            l = min(len(wf), N)
            l_s = l_s + l
            
            for j in range(l):
                eid = wf[j]['EventID']
                ch = wf[j]['ChannelID'] # in some Event, the amount of Channel < 30
                pe = answ.query('EventID=={} & ChannelID=={}'.format(eid, ch))
                pet = pe['PETime'].values # fetch corresponding PETime to the specific EventID & ChannelID
                unipe, c = np.unique(pet, return_counts=True)
                
                if np.size(unipe) == 1 and c[0] == 1: # if single pe
                    if unipe[0] < 21 or unipe[0] > 930:
                        print('opps! ' + str(eid)) # print Event when the single pe is too early or too late
                    else:
                        single_wf = wf[j]['Waveform'] # temporarily record waveform
                        dt['speWf'][num] = single_wf[unipe[0] - 1 - 20 : unipe[0] - 1 + 100] # only record 120 position, and the time of spe is the 21th
                        dt['TrainSet'][num] = i
                        dt['EventID'][num] = eid
                        dt['ChannelID'][num] = ch
                        dt['Waveform'][num] = single_wf
                        # The 21th position is the spe incoming time
                        num = num + 1 # preparing for next record
                    
                print('\rThe {}th Single PE Generating:|{}>{}|{:6.2f}%'.format(i, ((20*j)//l)*'-', (19 - (20*j)//l)*' ', 100 * ((j+1) / l)), end=''if j != l-1 else '\n') # show process bar

    dt = dt[np.where(dt['EventID'] > 0)] # cut empty dt part
    print('There are {} spe in {} waveforms'.format(num, l_s)) # show the amount of spe in l events
    '''
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 16 # set figure parameters
    
    spemean = np.mean(dt['speWf'], axis = 0) # calculate average fluctuation of waveform
    plt.figure()
    plt.xlim(0,120)
    plt.ylim(930, 980)
    tr = list(range(120))
    plt.plot(tr, spemean) # draw the average fluctuation
    plt.vlines([20], ymin=945, ymax=975)
    plt.xlabel('ns')
    plt.ylabel('mV')
    plt.title("Standard response model")
    plt.savefig('spemean.png')
    plt.close()
    
    spemin = np.min(dt['speWf'], axis = 1)
    u = np.unique(pet)
    plt.figure()
    plt.hist(spemin, len(u), density=1, histtype='bar', cumulative=False) # show the dispersion of minimum of spe waveform
    plt.xlabel('mV')
    plt.savefig('specumu.png')
    plt.close()
    '''
    with h5py.File(single_pe_path, "w") as spp:
        spp.create_dataset('Sketchy', data=dt, compression='gzip') # save the spe events

def main(h5_path, single_pe_path):
    if not os.path.exists(single_pe_path):
        generate_standard(h5_path, single_pe_path) # generate response model

if __name__ == '__main__':
    main(args.ipt, args.opt)
