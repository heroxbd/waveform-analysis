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
psr.add_argument('--ref', type=str, help='reference file')
psr.add_argument('--num', type=int, help='fragment number')
psr.add_argument('-p', dest='print', action='store_false', help='print bool', default=True)
args = psr.parse_args()

if args.print:
    sys.stdout = None

def main(fopt, fipt, single_pe_path, method):
    spe_pre = wff.read_model(single_pe_path)
    opdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint32), ('PETime', np.uint16), ('Weight', np.float16)])
    with h5py.File(fipt, 'r', libver='latest', swmr=True) as ipt:
        ent = ipt['Waveform']
        lenfr = math.floor(len(ent)/(args.num+1))
        num = int(re.findall(r'-\d+\.h5', fopt, flags=0)[0][1:-3])
        if (num+2)*lenfr > len(ent):
            l = len(ent) - num*lenfr
        else:
            l = num*lenfr
        print('{} waveforms will be computed'.format(l))

        leng = len(ent[0]['Waveform'])
        assert leng >= len(spe_pre['spe']), 'Single PE too long which is {}'.format(len(spe_pre['spe']))
        dt = np.zeros(l * (leng//5), dtype=opdt)
        start = 0
        end = 0
        gen = wff.Ind_Generator()
        
        for i in range(l):
            wave = wff.deduct_base(spe_pre['epulse'] * ent[num*lenfr+i]['Waveform'], spe_pre['m_l'], spe_pre['thres'], 20, 'detail')

            if method == 'xiaopeip':
                pf = wff.fit_N(wave, spe_pre, 'xiaopeip')
            elif method == 'lucyddm':
                pf = wff.lucyddm_core(wave, spe_pre['spe'])
            elif method == 'mcmc':
                pf = wff.fit_N(wave, spe_pre, 'mcmc', gen)
            pet, pwe = wff.pf_to_tw(pf, 0.01)

            lenpf = pwe.shape[0]
            end = start + lenpf
            dt['PETime'][start:end] = pet.astype(np.uint16)
            dt['Weight'][start:end] = pwe.astype(np.float16)
            dt['EventID'][start:end] = ent[i]['EventID']
            dt['ChannelID'][start:end] = ent[i]['ChannelID']
            start = end
            print('\rAnsw Generating:|{}>{}|{:6.2f}%'.format(((20*i)//l)*'-', (19-(20*i)//l)*' ', 100 * ((i+1) / l)), end='' if i != l-1 else '\n')
    dt = dt[dt['Weight'] > 0]
    dt = np.sort(dt, kind='stable', order=['EventID', 'ChannelID', 'PETime'])
    with h5py.File(fopt, 'w') as opt:
        opt.create_dataset('Answer', data=dt, compression='gzip')
        print('The output file path is {}'.format(fopt), end=' ', flush=True)
    return

if __name__ == '__main__':
    main(args.opt, args.ipt, args.ref, args.met)