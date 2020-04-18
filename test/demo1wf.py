# -*- coding: utf-8 -*-

import re
import numpy as np
import scipy.stats
import h5py
import sys
sys.path.append('test')
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
psr.add_argument('-o', dest='opt', type=str, help='output file')
psr.add_argument('ipt', type=str, help='input file')
psr.add_argument('--met', type=str, help='fitting method')
psr.add_argument('--ref', type=str, help='reference file')
psr.add_argument('--event', '-e', type=int, dest='ent')
psr.add_argument('--channel', '-c', type=int, dest='cha')
psr.add_argument('--save', dest='save', action='store_true', help='save result to h5 file, must include -o argument', default=False)
args = psr.parse_args()

def main(fopt, fipt, single_pe_path, method):
    spe_pre = wff.read_model(single_pe_path)
    print('spe is {}'.format(spe_pre['spe']))
    opdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint32), ('PETime', np.uint16), ('Weight', np.float16)])
    with h5py.File(fipt, 'r', libver='latest', swmr=True) as ipt:
        ent = ipt['Waveform']
        leng = len(ent[0]['Waveform'])
        Chnum = len(np.unique(ent['ChannelID']))
        a1 = max(args.ent*Chnum-10000, 0)
        a2 = min(args.ent*Chnum+10000, len(ent))
        ent = ent[a1:a2]
        assert leng >= len(spe_pre['spe']), 'Single PE too long which is {}'.format(len(spe_pre['spe']))
        i = np.where(np.logical_and(ent['EventID'] == args.ent, ent['ChannelID'] == args.cha))[0][0]

        wave = wff.deduct_base(ent[i]['Waveform'], spe_pre['epulse'], spe_pre['m_l'], spe_pre['thres'], 20, 'detail')

        if method == 'xiaopeip':
                pf = wff.fit_N(wave, spe_pre, 'xiaopeip')
        elif method == 'lucyddm':
                pf = wff.lucyddm_core(wave, spe_pre['spe'])
        elif method == 'mcmc':
                pf = wff.fit_N(wave, spe_pre, 'mcmc', gen)
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

        pet_a, pwe_a = ad.xpp_convol(pet, pwe)
        dt_a = np.zeros(len(pet_a), dtype=opdt)
        dt_a['PETime'] = pet_a
        dt_a['Weight'] = pwe_a
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
            plt.hlines(spe_pre['thres'], 200, 500, color='c')
            t, c = np.unique(tru_pet, return_counts=True)
            plt.vlines(t, 0, 200*c, color='k')
            plt.vlines(pet_a, -200*pwe_a, 0, color='m')
            hh = -200*(np.max(pwe_a)+1)
            plt.vlines(pet, -200*pwe+hh, hh, color='y')
            plt.savefig('demo.png')
            plt.close()
            plt.plot(spe_r, c='b')
            plt.grid()
            plt.xlabel(r'Time/[ns]')
            plt.ylabel(r'ADC')
            plt.savefig('spe.png')
            plt.close()

if __name__ == '__main__':
    main(args.opt, args.ipt, args.ref, args.met)