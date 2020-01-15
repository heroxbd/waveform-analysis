# -*- coding: utf-8 -*-

#from memory_profiler import profile
import numpy as np
from scipy import optimize as opti
import h5py
import time
import matplotlib.pyplot as plt
import argparse
import sys
import gc

psr = argparse.ArgumentParser()
psr.add_argument('-a', dest='at', type=int)
psr.add_argument('-b', dest='bt', type=int)
args = psr.parse_args()

def norm_fit(x, M, p):
    return np.linalg.norm(p - np.matmul(M, x))

@profile
def main(a, b, fipt='ztraining-0.h5', aver_spe_path='test/xiaopeip/averspe.h5'):
    speFile = h5py.File(aver_spe_path, 'r', libver='latest', swmr=True)
    spemean = np.array(speFile['spe'])

    with h5py.File(fipt, 'r', libver='latest', swmr=True) as ipt:
        ent = ipt['Waveform']
        for i in range(a,b,1):
            wf_input = ent[i]['Waveform']
            core(wf_input, spemean)
    return 

#@profile
def core(wf_input, spemean):
    wave = wf_input - np.mean(wf_input[900:1000])
    lowp = np.argwhere(wave < -6.5).flatten()
    flag = 1
    lowp = lowp[np.logical_and(lowp>=2, lowp<1028)]
    if len(lowp) != 0:
        panel = np.zeros(1029)
        for j in lowp:
            head = j-7 if j-7 > 0 else 0
            tail = j+15+1 if j+15+1 <= 1029 else 1029
            panel[head:tail] = 1
        nihep = np.argwhere(panel == 1).flatten()
        xuhao = np.argwhere(wave[lowp+1]-wave[lowp]-wave[lowp-1]+wave[lowp-2] > 1.5).flatten()
        if len(xuhao) != 0:
            possible = np.unique(np.concatenate((lowp[xuhao]-10,lowp[xuhao]-9,lowp[xuhao]-8)))
            possible = possible[np.logical_and(possible>=0, possible<1029)]
            if len(possible) != 0:
                ans0 = np.zeros_like(possible).astype(np.float64)
                b = np.zeros((len(possible), 2))
                b[:, 1] = np.inf
                mne = spemean[np.mod(nihep.reshape(len(nihep), 1) - possible.reshape(1, len(possible)), 1029)]
                #ans = opti.fmin_l_bfgs_b(norm_fit, ans0, args=(mne, wave[nihep]), approx_grad=True, bounds=b)
                #ans = opti.fmin_slsqp(norm_fit, ans0, args=(mne, wave[nihep]), bounds=b, iprint=-1)
                ans = opti.fmin_tnc(norm_fit, ans0, args=(mne, wave[nihep]), approx_grad=True, bounds=b, messages=0)
                #print(sys.getsizeof(ans))
                pf = ans[0]
                #pf = ans
                del ans
                gc.collect()
            else:
                flag = 0
        else:
            flag = 0
    else:
        flag = 0
    if flag == 0:
        t = np.where(wave == wave.min())[0][:1] - np.argmin(spemean)
        possible = t if t[0] >= 0 else np.array([0])
        pf = np.array([1])
    pf[pf < 0.1] = 0
    lenpf = np.size(np.where(pf > 0))
    pet = possible[pf > 0]
    pwe = pf[pf > 0]
    pwe = pwe.astype(np.float16)

if __name__ == '__main__':
    main(args.at, args.bt)
    #main()
