# -*- coding: utf-8 -*-

import sys
import re
import numpy as np
import h5py
import math
import argparse
import wf_func as wff

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', type=str, help='output file')
psr.add_argument('ipt', type=str, help='input file')
psr.add_argument('--met', type=str, help='fitting method')
psr.add_argument('--ref', type=str, nargs='+', help='reference file')
psr.add_argument('--num', type=int, help='fragment number')
args = psr.parse_args()

def main(fopt, fipt, reference, method):
    if method == 'mcmc':
        sm = pickle.load(open(reference[1], 'rb'))
    spe_pre = wff.read_model(reference[0])
    opdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint32), ('PETime', np.uint16), ('Weight', np.float16)])
    with h5py.File(fipt, 'r', libver='latest', swmr=True) as ipt:
        ent = ipt['Waveform']
        lenfr = math.ceil(len(ent)/(args.num+1))
        num = int(re.findall(r'-\d+\.h5', fopt, flags=0)[0][1:-3])
        if (num+1)*lenfr > len(ent):
            l = len(ent) - num*lenfr
        else:
            l = lenfr
        print('{} waveforms will be computed'.format(l))

        leng = len(ent[0]['Waveform'])
        assert leng >= len(spe_pre['spe']), 'Single PE too long which is {}'.format(len(spe_pre['spe']))
        dt = np.zeros(l * (leng//5), dtype=opdt)
        start = 0
        end = 0
        
        for i in range(num*lenfr, num*lenfr+l):
            wave = wff.deduct_base(spe_pre['epulse'] * ent[i]['Waveform'], spe_pre['m_l'], spe_pre['thres'], 20, 'detail')

            if method == 'xiaopeip':
                pf = wff.fit_N(wave, spe_pre, 'xiaopeip')
            elif method == 'lucyddm':
                pf = wff.lucyddm_core(wave, spe_pre['spe'])
            elif method == 'mcmc':
                pf = wff.mcmc_core(wave, spe_pre)
            pet, pwe = wff.pf_to_tw(pf, 0.01)

            lenpf = pwe.shape[0]
            end = start + lenpf
            dt['PETime'][start:end] = pet.astype(np.uint16)
            dt['Weight'][start:end] = pwe.astype(np.float16)
            dt['EventID'][start:end] = ent[i]['EventID']
            dt['ChannelID'][start:end] = ent[i]['ChannelID']
            start = end
            # print('\rAnsw Generating:|{}>{}|{:6.2f}%'.format(((20*i)//l)*'-', (19-(20*i)//l)*' ', 100 * ((i+1) / l)), end='' if i != l-1 else '\n')
    dt = dt[dt['Weight'] > 0]
    dt = np.sort(dt, kind='stable', order=['EventID', 'ChannelID', 'PETime'])
    with h5py.File(fopt, 'w') as opt:
        opt.create_dataset('Answer', data=dt, compression='gzip')
        print('The output file path is {}'.format(fopt), end=' ', flush=True)
    return

if __name__ == '__main__':
    main(args.opt, args.ipt, args.ref, args.met)
