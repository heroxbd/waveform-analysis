#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 19:51:47 2019

@author: xudachengthu

Generate standard response model of single pe's waveform
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import time

h5_path = '/home/xudacheng/Downloads/GHdataset/finalcontest_data/ztraining-0.h5'
single_pe_path = '/home/xudacheng/Downloads/GHdataset/sketchystore/single_pe.h5'

def generate_standard(h5_path, single_pe_path):
    npdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint8), ('Waveform', np.uint16, 1029), ('speWf', np.uint16, 120)]) # set datatype
    
    ztrfile = h5py.File(h5_path) # read h5 file
    
    wf = ztrfile['Waveform'] # read waveform only
    answ = pd.read_hdf(h5_path, "GroundTruth") # read h5 file answer
    l = min(len(wf), 10000) # limit l to below 5, l is the amount of event
    wf = wf[0 : l]
    answ = answ[0 : 20*l] # assume 1 waveform has less than 20 answers
    dt = np.zeros(int(l/10), dtype = npdt) # assume 10 Events has less than 1 single pe event among them
    num = 0
    
    for i in range(l):
        eid = wf[i]['EventID']
        ch = wf[i]['ChannelID'] # in some Event, the amount of Channel < 30
        pe = answ.query("EventID=={} & ChannelID=={}".format(eid, ch))
        pet = pe['PETime'].values # fetch corresponding PETime to the specific EventID & ChannelID
        unipe, c = np.unique(pet, return_counts=True)
        
        if np.size(unipe) == 1 and c[0] == 1: # if single pe
            if unipe[0] < 21 or unipe[0] > 930:
                print('opps! ' + str(eid)) # print Event when the single pe is too early or too late
            else:
                spe_wf = wf[i]['Waveform'] # temporarily record waveform
                dt['speWf'][num] = spe_wf[unipe[0] - 1 - 20 : unipe[0] - 1 + 100] # only record 120 position, and the time of spe is the 21th
                dt['EventID'][num] = eid
                dt['ChannelID'][num] = ch
                dt['Waveform'][num] = spe_wf
                # The 21th position is the spe incoming time
                num = num + 1 # preparing for next record
            
        print("\rProcess:|{}>{}|{:6.2f}%".format(int((20*i)/l)*'-', (19 - int((20*i)/l))*' ', 100 * ((i+1) / l)), end='') # show process bar
    print('\n')
    
    dt = dt[np.where(dt['EventID'] > 0)] # cut empty dt part
    print('There are {} spe in {} waveforms'.format(num, l)) # show the amount of spe in l events
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
    spp = h5py.File(single_pe_path, "w")
    spp.create_dataset('Sketchy', data=dt, compression='gzip') # save the spe events

def main():
    start_t = time.time()
    generate_standard(h5_path, single_pe_path) # generate response model
    end_t = time.time()
    print('The total time is {}'.format(end_t - start_t))

if __name__ == '__main__':
    main()