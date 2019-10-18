#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 16:33:11 2019

@author: xudachengthu

using fft & ifft method and standard response model to deconvolution the waveform
"""

Length_pe = 1029

KNIFE = 0.05

AXE = 4

EXP = 4

import numpy as np
import h5py
import time
import standard
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft

fipt = "/home/xudacheng/Downloads/GHdataset/playground/playground-data.h5"
fopt = "/home/xudacheng/Downloads/GHdataset/playground/first-submission-spe.h5"

def generate_eff_ft():
    opdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint8), ('PETime', np.uint16), ('Weight', np.float16)])
    model = generate_model(standard.single_pe_path) # extract the model
    
    model = np.where(model > AXE, model - AXE, 0) # cut off unnecessary part to reduce statistical fluctuation
    
    core = model / np.max(model)
    for i in range(len(core)):
        core[i] = pow(core[i], EXP) # compress the model
    model = core * np.max(model) # the maximum height of model is unchanged
    model = np.where(model > 0.02, model, 0) # cut off all small values
    
    model_raw = np.concatenate([model, np.zeros(Length_pe - len(model))]) # concatenate the model
    
    model_k = fft(model_raw)
    
    with h5py.File(fipt, 'r') as ipt, h5py.File(fopt, "w") as opt:
        ent = ipt['Waveform']
        l = len(ent)
        print('{} waveforms will be computed'.format(l))
        dt = np.zeros(l * Length_pe, dtype = opdt)
        start = 0
        end = 0
        for i in range(l):
            wf_input = ent[i]['Waveform']
            wf_input = np.mean(wf_input[900:1000]) - wf_input # baseline reverse
            wf_input = np.where(wf_input > 0, wf_input, 0) # cut off all negative values
            wf_input = np.where(wf_input > AXE, wf_input - AXE, 0) # corresponding AXE cut
            wf_k = fft(wf_input) # fft for waveform input
            spec = np.divide(wf_k, model_k) # divide for deconvolution
            pf = ifft(spec)
            pf = pf.real
            
            pf = np.where(pf > KNIFE, pf, 0) # cut off all small values
            lenpf = np.size(np.where(pf > 0))
            if lenpf == 0:
                pf[300] = 1 # when there is no prediction of single pe, assume the 301th is single pe
            
            lenpf = np.size(np.where(pf > 0))
            pet = np.where(pf > 0)[0] # count the pe time
            pwe = pf[pf > 0] # calculate the pe weight
            pwe = pwe.astype(np.float16)
            end = start + lenpf
            dt['PETime'][start:end] = pet
            dt['Weight'][start:end] = pwe
            dt['EventID'][start:end] = ent[i]['EventID']
            dt['ChannelID'][start:end] = ent[i]['ChannelID'] # integrated saving related information
            start = end
            
            print("\rProcess:|{}>{}|{:6.2f}%".format(int((20*i)/l)*'-', (19 - int((20*i)/l))*' ', 100 * ((i+1) / l)), end='') # show process bar
        print('\n')
        dt = dt[np.where(dt['Weight'] > 0)] # cut empty dt part
        opt.create_dataset('Answer', data = dt, compression='gzip')
        print('The output file path is {}'.format(fopt), end = ' ', flush=True)

def generate_model(spe_path):
    speFile = h5py.File(spe_path, 'r')
    spemean = np.mean(speFile['Sketchy']['speWf'], axis = 0)
    base_vol = np.mean(spemean[70:120])
    stdmodel = base_vol - spemean[20:120] # stdmodel[0] is the single pe's incoming time & baseline inverse
    #stdmodel = np.around(stdmodel / 0.05) * 0.05 # smooth the stdmodel
    stdmodel = np.where(stdmodel > 0.02, stdmodel, 0) # cut off all small values
    stdmodel = np.where(stdmodel >= 0, stdmodel, 0) # cut off all negative values
    speFile.close()
    return stdmodel

def main():
    start_t = time.time()
    generate_eff_ft()
    end_t = time.time()
    print('The total time is {}'.format(end_t - start_t))

if __name__ == '__main__':
    main()