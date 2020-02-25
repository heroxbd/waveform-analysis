# -*- coding: utf-8 -*-

import re
import numpy as np
from scipy import optimize as opti
import scipy.stats
import h5py
import sys
sys.path.append('test')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import wf_analysis_func as wfaf

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 8
plt.rcParams['lines.markersize'] = 4.0
plt.rcParams['lines.linewidth'] = 1.0

psr = argparse.ArgumentParser()
psr.add_argument('ipt', help='input file')
psr.add_argument('-o', dest='opt', help='output file')
psr.add_argument('--event', '-e', type=int, dest='ent')
psr.add_argument('--channel', '-c', type=int, dest='cha')
psr.add_argument('--save', dest='save', action='store_true', help='save result to h5 file, must include -o argument', default=False)
args = psr.parse_args()

single_pe_path = 'xtest/averspe.h5'

def norm_fit(x, M, p):
    return np.linalg.norm(p - np.matmul(M, x))

def main():
    fipt = args.ipt
    fopt = args.opt
    epulse = wfaf.estipulse(args.ipt)
    spemean_r = wfaf.generate_model(single_pe_path, epulse)
    spemean = -1 * epulse * spemean_r
    print('spemean is {}'.format(spemean))
    peak_c, zero_l, m_l, mar_l, mar_r, thres = wfaf.pre_analysis(fipt, epulse, spemean)
    opdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint32), ('PETime', np.uint16), ('Weight', np.float16)])
    with h5py.File(fipt, 'r', libver='latest', swmr=True) as ipt:
        ent = ipt['Waveform']
        Length_pe = len(ent[0]['Waveform'])
        dt = np.zeros(Length_pe, dtype=opdt)
        Chnum = len(np.unique(ent['ChannelID']))
        a1 = max(args.ent*Chnum-10000, 0)
        a2 = min(args.ent*Chnum+10000, len(ent))
        ent = ent[a1:a2]
        assert Length_pe >= len(spemean), 'Single PE too long which is {}'.format(len(spemean))
        spemean = np.concatenate([spemean, np.zeros(Length_pe - len(spemean))])
        i = np.where(np.logical_and(ent['EventID'] == args.ent, ent['ChannelID'] == args.cha))[0][0]
        wf_input = ent[i]['Waveform']
        wf_input = -1 * epulse * wf_input
        #wave = wf_input - wfaf.find_base_fast(wf_input)
        wave = wf_input - wfaf.find_base(wf_input, m_l, thres)
        #lowp = np.argwhere(wfaf.vali_base(wave, m_l, thres) == 1).flatten()
        lowp = np.argwhere(wave < thres).flatten()
        flag = 1
        lowp = lowp[np.logical_and(lowp > 1, lowp < Length_pe-1)]
        if len(lowp) != 0:
            panel = np.zeros(Length_pe)
            for j in lowp:
                head = j-mar_l if j-mar_l > 0 else 0
                tail = j+mar_r+1 if j+mar_r+1 <= Length_pe else Length_pe
                panel[head:tail] = 1
            nihep = np.argwhere(panel == 1).flatten()
            xuhao = np.argwhere(wave[lowp+1]-wave[lowp]-wave[lowp-1]+wave[lowp-2] > 1.5).flatten()
            if len(xuhao) != 0:
                possible = np.unique(np.concatenate((lowp[xuhao]-(peak_c+2), lowp[xuhao]-(peak_c+1), lowp[xuhao]-peak_c, lowp[xuhao]-(peak_c-1))))
                possible = possible[np.logical_and(possible >= 0, possible < Length_pe)]
                if len(possible) != 0:
                    ans0 = np.zeros_like(possible).astype(np.float64)
                    b = np.zeros((len(possible), 2))
                    b[:, 1] = np.inf
                    #b[:, 1] = 5
                    mne = spemean[np.mod(nihep.reshape(len(nihep), 1) - possible.reshape(1, len(possible)), Length_pe)]
                    print('mne is {}, its shape is {}'.format(mne, mne.shape))
                    ans = opti.fmin_l_bfgs_b(norm_fit, ans0, args=(mne, wave[nihep]), approx_grad=True, bounds=b, maxfun=100000)
                    #ans = opti.fmin_slsqp(norm_fit, ans0, args=(mne, wave[nihep]), bounds=b, iprint=-1)
                    #ans = opti.fmin_tnc(norm_fit, ans0, args=(mne, wave[nihep]), approx_grad=True, bounds=b, messages=0, maxfun=10000)
                    print('ans is {}'.format(ans))
                    pf = ans[0]
                    if np.sum(pf <= 0.1) == len(pf):
                        flag = 0
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
        pwe = pf[pf > 0.1]
        pwe = pwe.astype(np.float16)
        lenpf = len(pwe)
        pet = possible[pf > 0.1]
        print('PETime = {}, Weight = {}'.format(pet, pwe))
        dt['PETime'][0:lenpf] = pet
        dt['Weight'][0:lenpf] = pwe
        dt['EventID'][0:lenpf] = args.ent
        dt['ChannelID'][0:lenpf] = args.cha
        dt = dt[dt['Weight'] > 0]
        dt = np.sort(dt, kind='stable', order=['EventID', 'ChannelID', 'PETime'])
        print('dt is {}'.format(dt))
        tth = ipt['GroundTruth']
        b = min((args.ent+1)*30*Chnum, len(tth))
        tth = tth[0:b]
        j = np.where(np.logical_and(tth['EventID'] == args.ent, tth['ChannelID'] == args.cha))
        print('PEnum is {}'.format(len(j[0])))
        tru_pet = tth[j]['PETime']
        print('The truth is {}'.format(np.sort(tru_pet)))
        wdist = scipy.stats.wasserstein_distance(tru_pet, pet, v_weights=pwe)
        Q = len(j[0]); q = np.sum(pwe)
        pdist = np.abs(Q - q) * scipy.stats.poisson.pmf(Q, Q)
        print('wdist is {}, pdist is {}'.format(wdist, pdist))
        if args.save:
            with h5py.File(fopt, 'w') as opt:
                dset = opt.create_dataset('Answer', data=dt, compression='gzip')
            plt.plot(wave, c='b')
            plt.scatter(nihep, wave[nihep], marker='x', c='g')
            plt.scatter(possible, wave[possible], marker='+', c='r')
            plt.grid()
            plt.xlabel(r'Time/[ns]')
            plt.ylabel(r'ADC')
            plt.xlim(200, 500)
            plt.hlines(thres, 200, 500, color='c')
            t, c = np.unique(tru_pet, return_counts=True)
            plt.vlines(t, -200*c, 0, color='m')
            plt.vlines(pet, 0, 200*pwe, color='k')
            plt.savefig('demo.png')
            plt.close()
            plt.plot(spemean_r, c='b')
            plt.grid()
            plt.xlabel(r'Time/[ns]')
            plt.ylabel(r'ADC')
            plt.savefig('spemean.png')
            plt.close()

if __name__ == '__main__':
    main()
