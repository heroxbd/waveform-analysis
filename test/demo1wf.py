# -*- coding: utf-8 -*-

import re
import pickle
import numpy as np
import scipy.stats
import h5py
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import wf_func as wff

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 8
plt.rcParams['lines.markersize'] = 4.0
plt.rcParams['lines.linewidth'] = 1.0

psr = argparse.ArgumentParser()
psr.add_argument('ipt', type=str, help='input file')
psr.add_argument('--met', type=str, help='fitting method')
psr.add_argument('--ref', type=str, nargs='+', help='reference file')
psr.add_argument('--event', '-e', type=int, dest='ent')
psr.add_argument('--channel', '-c', type=int, dest='cha')
psr.add_argument('--save', dest='save', action='store_true', help='save demo to png', default=False)
args = psr.parse_args()

def main(fipt, reference, method):
    if method == 'mcmc':
        sm = pickle.load(open(reference[1], 'rb'))
    spe_pre = wff.read_model(reference[0])
    print('spe is {}'.format(spe_pre['spe']))
    opdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint32), ('PETime', np.uint16), ('Weight', np.float16)])
    with h5py.File(fipt, 'r', libver='latest', swmr=True) as ipt:
        ent = ipt['Waveform']
        leng = len(ent[0]['Waveform'])
        Chnum = np.max(ent[max(-10000,-len(ent)):]['ChannelID'])
        a1 = max(args.ent*Chnum-10000, 0)
        a2 = min(args.ent*Chnum+10000, len(ent))
        ent = ent[a1:a2]
        assert leng >= len(spe_pre['spe']), 'Single PE too long which is {}'.format(len(spe_pre['spe']))
        v = np.logical_and(ent['EventID'] == args.ent, ent['ChannelID'] == args.cha)
        assert np.sum(v) > 0, 'Can not find in EventID & ChannelID'
        i = np.where(v)[0][0]

        wave = wff.deduct_base(spe_pre['epulse'] * ent[i]['Waveform'], spe_pre['m_l'], spe_pre['thres'], 20, 'detail')

        if method == 'xiaopeip':
                pf, fitp, possible = wff.fit_N(wave, spe_pre, 'xiaopeip', return_position=True)
        elif method == 'lucyddm':
                pf = wff.lucyddm_core(wave, spe_pre['spe'])
                fitp = []
                possible = []
        elif method == 'mcmc':
                pf, fitp, possible = wff.fit_N(wave, spe_pre, 'mcmc', model=sm, return_position=True)
        pet, pwe = wff.pf_to_tw(pf, 0.01)

        print('PETime = {}, Weight = {}'.format(pet, pwe))
        lenpf = len(pwe)
        dt = np.zeros(lenpf, dtype=opdt)
        dt['PETime'] = pet.astype(np.uint16)
        dt['Weight'] = pwe.astype(np.float16)
        dt['EventID'] = args.ent
        dt['ChannelID'] = args.cha
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
        pf1 = np.zeros(leng); pf1[pet] = pwe
        wave1 = np.convolve(spe_pre['spe'], pf1, 'full')[:leng-len(spe_pre['spe'])+1]
        print('Resi-norm = {}'.format(np.linalg.norm(wave-wave1)))

        pet_a, pwe_a = wff.xpp_convol(pet, pwe)
        dt_a = np.zeros(len(pet_a), dtype=opdt)
        dt_a['PETime'] = pet_a
        dt_a['Weight'] = pwe_a
        dt_a['EventID'] = args.ent
        dt_a['ChannelID'] = args.cha
        wdist_a = scipy.stats.wasserstein_distance(tru_pet, pet_a, v_weights=pwe_a)
        q_a = np.sum(pwe_a)
        pdist_a = np.abs(Q - q_a) * scipy.stats.poisson.pmf(Q, Q)
        print('wdist is {}, pdist is {}'.format(wdist_a, pdist_a))
        pf_a = np.zeros(leng); pf_a[pet] = pwe_a
        wave_a = np.convolve(spe_pre['spe'], pf_a, 'full')[:leng-len(spe_pre['spe'])+1]
        print('Resi-norm = {}'.format(np.linalg.norm(wave-wave_a)))

        if args.save:
            plt.plot(wave, c='b', label='original WF')
            plt.plot(wave1, c='y', label='before xpp_convol WF')
            plt.plot(wave_a, c='m', label='after xpp_convol WF')
            plt.scatter(fitp, wave[fitp], marker='x', c='g')
            plt.scatter(possible, wave[possible], marker='+', c='r')
            plt.grid()
            plt.xlabel(r'Time/[ns]')
            plt.ylabel(r'ADC')
            plt.xlim(200, 400)
            plt.hlines(spe_pre['thres'], 200, 500, color='c')
            t, c = np.unique(tru_pet, return_counts=True)
            plt.vlines(t, 0, 10*c, color='k')
            hh = -10*(np.max(pwe_a)+1)
            plt.vlines(pet, -10*pwe+hh, hh, color='y', label='before xpp_convol weight')
            plt.vlines(pet_a, -10*pwe_a, 0, color='m', label='after xpp_convol weight')
            plt.legend()
            plt.savefig('demo.png')
            plt.close()
            plt.plot(spe_pre['spe'], c='b')
            plt.grid()
            plt.xlabel(r'Time/[ns]')
            plt.ylabel(r'ADC')
            plt.savefig('spe.png')
            plt.close()

if __name__ == '__main__':
    main(args.ipt, args.ref, args.met)
