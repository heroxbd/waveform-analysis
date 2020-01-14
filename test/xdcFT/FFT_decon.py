# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 16:33:11 2019

@author: xudachengthu

using fft & ifft method and standard response model to deconvolution the waveform
"""

import argparse
import numpy as np
import h5py
import sys
sys.path.append('test')
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
import wf_analysis_func as wfaf

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', help='output')
psr.add_argument('ipt', help='input')
psr.add_argument('--ref')
psr.add_argument('-k', dest='kni')
psr.add_argument('-a', dest='axe')
psr.add_argument('-e', dest='exp')
args = psr.parse_args()

KNIFE = args.kni
AXE = args.axe
EXP = args.exp

def generate_eff_ft(fopt, fipt, single_pe_path):
    epulse = wfaf.estipulse(fipt)
    model = wfaf.generate_model(single_pe_path, epulse)
    opdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint32), ('PETime', np.uint16), ('Weight', np.float16)])
    model = compr(model, EXP, AXE, epulse)

    with h5py.File(fipt, 'r', libver='latest', swmr=True) as ipt, h5py.File(fopt, 'w') as opt:
        ent = ipt['Waveform']
        Length_pe = len(ent['Waveform'][0])
        model_raw = np.concatenate([model, np.zeros(Length_pe - len(model))])  # concatenate the model
        model_k = fft(model_raw)
        l = len(ent)
        print('{} waveforms will be computed'.format(l))
        dt = np.zeros(l * Length_pe, dtype = opdt)
        start = 0
        end = 0
        for i in range(l):
            wf_input = ent[i]['Waveform']
            wf_input = preman(wf_input, AXE, epulse)
            wf_k = fft(wf_input)  # fft for waveform input
            spec = np.divide(wf_k, model_k) # divide for deconvolution
            pf = ifft(spec)
            pf = pf.real
            
            pf = np.where(pf > KNIFE, pf, 0) # cut off all small values
            lenpf = np.size(np.where(pf > 0))
            if lenpf == 0:
                if epulse == -1:
                    pf[np.where(wf_input == wf_input.min())[0][:1] - np.argmin(model)] == 1
                elif epulse == 1:
                    pf[np.where(wf_input == wf_input.max())[0][:1] - np.argmax(model)] == 1
            
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
            
            print('\rAnsw Generating:|{}>{}|{:6.2f}%'.format(((20*i)//l)*'-', (19-(20*i)//l)*' ', 100 * ((i+1) / l)), end='' if i != l-1 else '\n') # show process bar
        dt = dt[np.where(dt['Weight'] > 0)] # cut empty dt part
        dset = opt.create_dataset('Answer', data = dt, compression='gzip')
        dset.attrs['totalLength'] = l
        dset.attrs['spePath'] = single_pe_path
        print('The output file path is {}'.format(fopt), end = ' ', flush=True)

def compr(model, exp, axe, epulse):
    if epulse == -1:
        model = np.where(model < -1*axe, model + axe, 0)
        core = model / np.min(model)
        core = np.power(core, exp)
        model = core * np.min(model) # the maximum height of model is unchanged
        model = np.where(model > 0.02, model, 0) # cut off all small values
    elif epulse == 1:
        model = np.where(model > axe, model - axe, 0)
        core = model / np.max(model)
        core = np.power(core, exp)
        model = core * np.max(model)
        model = np.where(model > 0.02, model, 0)  # cut off all small values
    return model

def preman(wf, axe, epulse):
    wf = wf - np.mean(wf[-100:])
    if epulse == -1:
        wf = np.where(wf < 0, wf, 0)
        wf = np.where(wf < -1*axe, wf + axe, 0)
    elif epulse == 1:
        wf = np.where(wf > 0, wf, 0)
        wf = np.where(wf > axe, wf - axe,0)
    return wf

def main(fopt, fipt, single_pe_path):
    generate_eff_ft(fopt, fipt, single_pe_path)

if __name__ == '__main__':
    main(args.opt, args.ipt, args.ref)
