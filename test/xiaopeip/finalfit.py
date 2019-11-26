# -*- coding: utf-8 -*-

import numpy as np
import h5py
import time
# import standard
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
import argparse

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', help='output')
psr.add_argument('ipt', help='input')
psr.add_argument('--ref')
args = psr.parse_args()

def main(fopt, fipt, aver_spe_path):
    speFile = h5py.File(aver_spe_path, 'r', libver='latest', swmr=True)
    spemean = speFile['spe']
    aver = speFile['averzero']
    opdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint8), ('PETime', np.uint16), ('Weight', np.float16)])

    with h5py.File(fipt, 'r', libver='latest', swmr=True) as ipt, h5py.File(fopt, 'w') as opt:
        ent = ipt['Waveform']
        l = len(ent)
        print('{} waveforms will be computed'.format(l))
        dt = np.zeros(l * 1029, dtype=opdt)
        start = 0
        end = 0
        start_t = time.time()
        for i in range(l):
            wf_input = ent[i]['Waveform']
            wave = wf_input - 972 - aver
            lowp = np.argwhere(wave < -6.5)
            if len(lowp) != 0:
                if lowp[0] < 1:
                    lowp = lowp[1:]
                if lowp[-1] >= 1028:
                    lowp = lowp[:-1]
                panel = np.zeros(1029)
                ddiff = np.zeros(len(lowp))
                for j in lowp:
                    head = j-7 if j-7 > 0 else 0
                    tail = j+15+1 if j+15+1 <= 1029 else 1029
                    panel[head:tail] = 1
                    ddiff[j] = wave[j+1]-wave[j]-wave[j-1]+wave[j-2]
                nihep = np.argwhere(panel == 1)
                xuhao = lowp[ddiff > 1.5]
                
        print('\rAnsw Generating:|{}>{}|{:6.2f}%'.format(((20*i)//l)*'-', (19-(20*i)//l)*' ', 100 * ((i+1) / l)), end='' if i != l-1 else '\n') # show process bar
    end_t = time.time()
    return 

if __name__ == '__main__':
    main(args.opt, args.ipt, args.ref)
