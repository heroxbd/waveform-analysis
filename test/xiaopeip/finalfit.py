# -*- coding: utf-8 -*-

import re
import numpy as np
from scipy import optimize as opti
import h5py
import sys
sys.path.append('test')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import argparse
import wf_analysis_func as wfaf

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', help='output file')
psr.add_argument('ipt', help='input file')
psr.add_argument('--ref', dest='ref', help='reference file')
psr.add_argument('--num', type=int, help='fragment number')
psr.add_argument('-p', dest='print', action='store_false', help='print bool', default=True)
args = psr.parse_args()

if args.print:
    sys.stdout = None

def norm_fit(x, M, p):
    return np.linalg.norm(p - np.matmul(M, x))

def xiaopeip_N(wave, spe_pre, l):
    lowp = np.argwhere(wfaf.vali_base(wave, spe_pre['m_l'], spe_pre['thres']) == 1).flatten()
    lowp = lowp[np.logical_and(lowp > 1, lowp < l-1)]
    flag = 1
    if len(lowp) != 0:
        panel = np.zeros(l)
        for j in lowp:
            head = j-spe_pre['mar_l'] if j-spe_pre['mar_l'] > 0 else 0
            tail = j+spe_pre['mar_r']+1 if j+spe_pre['mar_r']+1 <= l else l
            panel[head:tail] = 1
        nihep = np.argwhere(panel == 1).flatten()
        xuhao = np.argwhere(wave[lowp+1]-wave[lowp]-wave[lowp-1]+wave[lowp-2] > 1.5).flatten()
        if len(xuhao) != 0:
            possible = np.unique(np.concatenate((lowp[xuhao]-(spe_pre['peak_c']+2), lowp[xuhao]-(spe_pre['peak_c']+1), lowp[xuhao]-spe_pre['peak_c'], lowp[xuhao]-(spe_pre['peak_c']-1))))
            possible = possible[np.logical_and(possible>=0, possible<l)]
            if len(possible) != 0:
                ans0 = np.zeros_like(possible).astype(np.float64)
                b = np.zeros((len(possible), 2)).astype(np.float64)
                b[:, 1] = np.inf
                mne = spe_pre['spemean'][np.mod(nihep.reshape(len(nihep), 1) - possible.reshape(1, len(possible)), l)]
                ans = opti.fmin_l_bfgs_b(norm_fit, ans0, args=(mne, wave[nihep]), approx_grad=True, bounds=b, maxfun=500000)
                # ans = opti.fmin_slsqp(norm_fit, ans0, args=(mne, wave[nihep]), bounds=b, iprint=-1, iter=500000)
                # ans = opti.fmin_tnc(norm_fit, ans0, args=(mne, wave[nihep]), approx_grad=True, bounds=b, messages=0, maxfun=500000)
                pf_r = ans[0]
                # print(ans[2]['warnflag'])
                if np.sum(pf_r <= 0.1) == len(pf_r):
                    flag = 0
            else:
                flag = 0
        else:
            flag = 0
    else:
        flag = 0
    if flag == 0:
        t = np.where(wave == wave.min())[0][:1] - np.argmin(spe_pre['spemean'])
        possible = t if t[0] >= 0 else np.array([0])
        pf_r = np.array([1])
    pf = np.zeros_like(wave)
    pf[possible] = pf_r
    return pf, nihep, possible

def main(fopt, fipt, single_pe_path):
    spemean, epulse = wfaf.generate_model(single_pe_path)
    spe_pre = wfaf.pre_analysis(fipt, epulse, -1*epulse*spemean)

    opdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint32), ('PETime', np.uint16), ('Weight', np.float16)])
    with h5py.File(fipt, 'r', libver='latest', swmr=True) as ipt:
        ent = ipt['Waveform']
        l = len(ent)
        print('{} waveforms will be computed'.format(l))
        lenfr = math.floor(len(ent)/(args.num+1))
        num = int(re.findall(r'-\d+\.h5', fopt, flags=0)[0][1:-3])
        if (num+2)*lenfr > len(ent):
            ent = ent[num*lenfr:]
        else:
            ent = ent[num*lenfr:(num+1)*lenfr]

        Length_pe = len(ent[0]['Waveform'])
        assert Length_pe >= len(spe_pre['spemean']), 'Single PE too long which is {}'.format(len(spe_pre['spemean']))
        spe_pre['spemean'] = np.concatenate([spe_pre['spemean'], np.zeros(Length_pe - len(spe_pre['spemean']))])
        dt = np.zeros(l * (Length_pe//5), dtype=opdt)
        start = 0
        end = 0
        for i in range(l):
            wf_input = ent[i]['Waveform']
            wave = -1*spe_pre['epulse']*wfaf.deduct_base(-1*spe_pre['epulse']*wf_input, spe_pre['m_l'], spe_pre['thres'], 10, 'detail')
            pf = xiaopeip_N(wave, spe_pre, Length_pe)
            pet, pwe = wfaf.pf_to_tw(pf, 0.1)

            lenpf = len(pwe)
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
    main(args.opt, args.ipt, args.ref)
