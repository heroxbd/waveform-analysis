# -*- coding: utf-8 -*-

import re
import numpy as np
from scipy import optimize as opti
import h5py
import sys
sys.path.append('test')
import matplotlib.pyplot as plt
import argparse
import wf_analysis_func as wfaf

psr = argparse.ArgumentParser()
psr.add_argument('ipt', help='input file')
psr.add_argument('-o', dest='opt', help='output file')
psr.add_argument('--event', '-e', type=int, dest='ent')
psr.add_argument('--channel', '-c', type=int, dest='cha')
psr.add_argument('--save', dest='save', action='store_true', help='save result to h5 file, must include -o argument', default=False)
args = psr.parse_args()

single_pe_path = 'test/xiaopeip/averspe.h5'

def norm_fit(x, M, p):
    return np.linalg.norm(p - np.matmul(M, x))

def main():
    epulse = -1
    spemean = wfaf.generate_model(single_pe_path, epulse)
    opdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint8), ('PETime', np.uint16), ('Weight', np.float16)])
    dt = np.zeros(1029, dtype=opdt)
    with h5py.File(args.ipt, 'r', libver='latest', swmr=True) as ipt:
        ent = ipt['Waveform']
        Chnum = len(np.unique(ent[0:args.ent*30]['ChannelID']))
        a1 = max(args.ent*Chnum-10000, 0)
        a2 = min(args.ent*Chnum+10000, len(ent))
        ent = ent[a1:a2]
        Length_pe = len(ent[0]['Waveform'])
        spemean = np.concatenate([spemean, np.zeros(Length_pe - len(spemean))])
        i = np.where(np.logical_and(ent['EventID'] == args.ent, ent['ChannelID'] == args.cha))[0][0]
        wf_input = ent[i]['Waveform']
        #plt.plot(wf_input)
        wave = wf_input - np.mean(wf_input[900:1000])
        lowp = np.argwhere(wave < -6.5).flatten()
        flag = 1
        lowp = lowp[np.logical_and(lowp > 1, lowp < Length_pe-1)]
        if len(lowp) != 0:
            panel = np.zeros(Length_pe)
            for j in lowp:
                head = j-7 if j-7 > 0 else 0
                tail = j+15+1 if j+15+1 <= Length_pe else Length_pe
                panel[head:tail] = 1
            nihep = np.argwhere(panel == 1).flatten()
            xuhao = np.argwhere(wave[lowp+1]-wave[lowp]-wave[lowp-1]+wave[lowp-2] > 1.5).flatten()
            if len(xuhao) != 0:
                possible = np.unique(np.concatenate((lowp[xuhao]-10, lowp[xuhao]-9, lowp[xuhao]-8,lowp[xuhao]-7)))
                possible = possible[np.logical_and(possible >= 0, possible < Length_pe)]
                if len(possible) != 0:
                    ans0 = np.zeros_like(possible).astype(np.float64)
                    b = np.zeros((len(possible), 2))
                    b[:, 1] = np.inf
                    #b[:, 1] = 5
                    mne = spemean[np.mod(nihep.reshape(len(nihep), 1) - possible.reshape(1, len(possible)), Length_pe)]
                    #ans = opti.fmin_l_bfgs_b(norm_fit, ans0, args=(mne, wave[nihep]), approx_grad=True, bounds=b)
                    #ans = opti.fmin_slsqp(norm_fit, ans0, args=(mne, wave[nihep]), bounds=b, iprint=-1)
                    ans = opti.fmin_tnc(norm_fit, ans0, args=(mne, wave[nihep]), approx_grad=True, bounds=b, maxfun=1000)
                    print('ans is {}'.format(ans))
                    pf = ans[0]
                    #pf = ans
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
        if np.sum(pf < 0.1) != len(pf):
            pf[pf < 0.1] = 0
        pwe = pf[pf > 0]
        pwe = pwe.astype(np.float16)
        lenpf = len(pwe)
        pet = possible[pf > 0]
        print('PETime = {}, Weight = {}'.format(pet, pwe))
        dt['PETime'][0:lenpf] = pet
        dt['Weight'][0:lenpf] = pwe
        dt['EventID'][0:lenpf] = args.ent
        dt['ChannelID'][0:lenpf] = args.cha
        dt = dt[dt['Weight'] > 0]
        dt = np.sort(dt, kind='stable', order=['EventID', 'ChannelID', 'PETime'])
        print('dt is {}'.format(dt))
        if args.save:
            with h5py.File(fopt, 'w') as opt:
                dset = opt.create_dataset('Answer', data=dt, compression='gzip')

if __name__ == '__main__':
    main()
