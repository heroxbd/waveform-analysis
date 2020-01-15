# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

def estipulse(h5_path):
    with h5py.File(h5_path, 'r', libver='latest', swmr=True) as ztrfile:
        Wf = ztrfile['Waveform']['Waveform']
        flag = 0
        i = 0
        while flag == 0:
            mm = np.max(Wf[i]) - np.min(Wf[i])
            mean = np.mean(Wf[i])
            if np.max(Wf[i]) - mean > 4/5*mm:
                flag = 1
            if mean - np.min(Wf[i]) > 4/5*mm:
                flag = -1
    return flag

def generate_model(spe_path, epulse):
    with h5py.File(spe_path, 'r', libver='latest', swmr=True) as speFile:
        spemean = np.mean(speFile['SinglePE']['speWf'], axis=0)
        base_vol = np.mean(spemean[-10:])
        # stdmodel[0] is the single pe's incoming time
        stdmodel = spemean[:-10] - base_vol
        #stdmodel = np.around(stdmodel / 0.05) * 0.05 # smooth the stdmodel
        # cut off all small values
        if epulse == -1:
            stdmodel = np.where(stdmodel < -0.02, stdmodel, 0)
        elif epulse == 1:
            stdmodel = np.where(stdmodel > 0.02, stdmodel, 0)
        #stdmodel = np.where(stdmodel >= 0, stdmodel, 0) # cut off all negative values
    return stdmodel

def speplot(dt):
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 16  # set figure parameters

    # calculate average fluctuation of waveform
    spemean = np.mean(dt['speWf'], axis=0)
    plt.figure()
    plt.xlim(0, 50)
    plt.plot(spemean)  # draw the average fluctuation
    plt.xlabel('ns')
    plt.ylabel('mV')
    plt.title("Standard response model")
    #plt.savefig('spemean.png')
    plt.close()

    spemin = np.min(dt['speWf'], axis=1)
    u = np.unique(Pt)
    plt.figure()
    # show the dispersion of minimum of spe waveform
    plt.hist(spemin, len(u), density=1, histtype='bar', cumulative=False)
    plt.xlabel('mV')
    #plt.savefig('specumu.png')
    plt.close()
