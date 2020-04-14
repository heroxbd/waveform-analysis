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
import finalfit as ff
import adjust as ad

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
    spemean_r, epulse = wfaf.generate_model(single_pe_path)
    spe_pre = wfaf.pre_analysis(fipt, epulse, -1*epulse*spemean_r)
    print('spemean is {}'.format(spe_pre['spemean']))
    opdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint32), ('PETime', np.uint16), ('Weight', np.float16)])
    with h5py.File(fipt, 'r', libver='latest', swmr=True) as ipt:
        ent = ipt['Waveform']
        Length_pe = len(ent[0]['Waveform'])
        dt = np.zeros(Length_pe, dtype=opdt)
        Chnum = len(np.unique(ent['ChannelID']))
        a1 = max(args.ent*Chnum-10000, 0)
        a2 = min(args.ent*Chnum+10000, len(ent))
        ent = ent[a1:a2]
        assert Length_pe >= len(spe_pre['spemean']), 'Single PE too long which is {}'.format(len(spe_pre['spemean']))
        i = np.where(np.logical_and(ent['EventID'] == args.ent, ent['ChannelID'] == args.cha))[0][0]
        spe_pre['spemean'] = np.concatenate([spe_pre['spemean'], np.zeros(Length_pe - len(spe_pre['spemean']))])

        wf_input = ent[i]['Waveform']
        wf_input = -1 * spe_pre['epulse'] * wf_input
        wave = -1*spe_pre['epulse']*wfaf.deduct_base(-1*spe_pre['epulse']*wf_input, spe_pre['m_l'], spe_pre['thres'], 10, 'detail')
        pf, nihep, possible = ff.xiaopeip_N(wave, spe_pre, Length_pe)
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
            plt.scatter(nihep, wave[nihep], marker='x', c='g')
            plt.scatter(possible, wave[possible], marker='+', c='r')
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
            plt.plot(spemean_r, c='b')
            plt.grid()
            plt.xlabel(r'Time/[ns]')
            plt.ylabel(r'ADC')
            plt.savefig('spemean.png')
            plt.close()

if __name__ == '__main__':
    main()
