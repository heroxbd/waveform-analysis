# -*- coding: utf-8 -*-

import re
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
                pf, possible = wff.mcmc_core(wave, spe_pre, return_position=True)
                fitp = []
        pet, pwe = wff.pf_to_tw(pf, 0.01)

        lenpf = len(pwe)
        dt = np.zeros(lenpf, dtype=opdt)
        dt['PETime'] = pet.astype(np.uint16)
        dt['Weight'] = pwe.astype(np.float16)
        dt['EventID'] = args.ent
        dt['ChannelID'] = args.cha
if __name__ == '__main__':
    main(args.ipt, args.ref, args.met)
