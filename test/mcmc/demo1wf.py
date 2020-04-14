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
import mcmcfit as mf

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

R = mf.R
N = mf.N

def main():
    fipt = args.ipt
    fopt = args.opt
    spemean, epulse = wfaf.generate_model(single_pe_path)
    _, _, m_l, _, _, thres = wfaf.pre_analysis(fipt, epulse, -1 * epulse * spemean)
    spemean = epulse * spemean
    print('spemean is {}'.format(spemean))
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
        i = np.where(np.logical_and(ent['EventID'] == args.ent, ent['ChannelID'] == args.cha))[0][0]

        sigma = (Length_pe - len(spemean) + 1) / (2*R)
        gen = mf.Ind_Generator(sigma)

        wf_input = ent[i]['Waveform']
        wave = epulse * wfaf.deduct_base(-1*epulse*wf_input, m_l, thres, 10, 'detail')
        pf = mf.mcmc_N(wave, spemean, gen)
        pet, pwe = wfaf.pf_to_tw(pf, 0.1)

        print('PETime = {}, Weight = {}'.format(pet, pwe))
        lenpf = len(pwe)
        dt['PETime'][:lenpf] = pet.astype(np.uint16)
        dt['Weight'][:lenpf] = pwe.astype(np.float16)
        dt['EventID'][:lenpf] = args.ent
        dt['ChannelID'][:lenpf] = args.cha
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

        idt = np.dtype([('PETime', np.int16), ('Weight', np.float16), ('Wgt_b', np.uint8)])
        seg = np.zeros(np.max(pet) + 3, dtype=idt)
        seg['PETime'] = np.arange(-1, np.max(pet) + 2)
        seg['Weight'][np.sort(pet) + 1] = pwe[np.argsort(pet)]
        seg['Wgt_b'] = np.around(seg['Weight'])
        resi = seg['Weight'][1:-1] - seg['Wgt_b'][1:-1]
        t = np.convolve(resi, [0.9, 1.7, 0.9], 'full')
        ta = np.diff(t, prepend=t[0])
        tb = np.diff(t, append=t[-1])
        seg['Wgt_b'][(ta > 0)*(tb < 0)*(t > 0.5)*(seg['Wgt_b'] == 0.0)*(seg['Weight'] > 0)] += 1
        if np.sum(seg['Wgt_b'][1:-1] > 0) != 0:
            wgt_a = seg['Wgt_b'][1:-1][seg['Wgt_b'][1:-1] > 0]
            pet_a = seg['PETime'][1:-1][seg['Wgt_b'][1:-1] > 0]
        else:
            wgt_a = 1
            pet_a = seg['PETime'][np.argmax(seg['Weight'])]
        dt_a = np.zeros(len(pet_a), dtype=opdt)
        dt_a['PETime'] = pet_a
        dt_a['Weight'] = wgt_a
        dt_a['EventID'] = args.ent
        dt_a['ChannelID'] = args.cha

        if args.save:
            with h5py.File(fopt, 'w') as opt:
                opt.create_dataset('Answer_un', data=dt, compression='gzip')
                opt.create_dataset('Answer', data=dt_a, compression='gzip')
            plt.plot(wave, c='b')
            plt.grid()
            plt.xlabel(r'Time/[ns]')
            plt.ylabel(r'ADC')
            plt.xlim(200, 500)
            plt.hlines(thres, 200, 500, color='c')
            t, c = np.unique(tru_pet, return_counts=True)
            plt.vlines(t, 0, 200*c, color='k')
            plt.vlines(pet_a, -200*wgt_a, 0, color='m')
            hh = -200*(np.max(wgt_a)+1)
            plt.vlines(pet, -200*pwe+hh, hh, color='y')
            plt.savefig('demo.png')
            plt.close()
            plt.plot(spemean, c='b')
            plt.grid()
            plt.xlabel(r'Time/[ns]')
            plt.ylabel(r'ADC')
            plt.savefig('spemean.png')
            plt.close()

if __name__ == '__main__':
    main()
