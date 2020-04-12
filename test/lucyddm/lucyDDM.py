# -*- coding: utf-8 -*-

import sys
sys.path.append('test')
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import wf_analysis_func as wfaf

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', help='output file')
psr.add_argument('ipt', help='input file')
psr.add_argument('--ref', dest='ref', help='reference file')
psr.add_argument('-p', dest='print', action='store_false', help='print bool', default=True)
args = psr.parse_args()

def lucyDDM(waveform, spe, iterations=50):
    '''Lucy deconvolution
    Parameters
    ----------
    waveform : 1d array
    spe : 1d array
        point spread function; single photon electron response
    iterations : int

    Returns
    -------
    signal : 1d array

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
    .. [2] https://github.com/scikit-image/scikit-image/blob/master/skimage/restoration/deconvolution.py#L329
    '''
    waveform = waveform.astype(np.float)
    spe = spe.astype(np.float)
    waveform = waveform / np.sum(spe)
    # use the deconvlution method
    wave_deconv = np.full(waveform.shape, 0.5)
    spe_mirror = spe[::-1]
    for _ in range(iterations):
        relative_blur = waveform / np.convolve(wave_deconv, spe, 'same')
        wave_deconv = wave_deconv * np.convolve(relative_blur, spe_mirror, 'same')
        # there is no need to set the bound if the spe and the wave are all none negative 
    return wave_deconv

def main(fopt, fipt, single_pe_path):
    epulse = wfaf.estipulse(fipt)
    spemean = wfaf.generate_model(single_pe_path, epulse)
    spemean = -1 * epulse * spemean
    _, _, m_l, _, _, thres = wfaf.pre_analysis(fipt, epulse, spemean)
    opdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint32), ('PETime', np.uint16), ('Weight', np.float16)])
    with h5py.File(fipt, 'r', libver='latest', swmr=True) as ipt:
        ent = ipt['Waveform']
        Length_pe = len(ent[0]['Waveform'])
        assert Length_pe >= len(spemean), 'Single PE too long which is {}'.format(len(spemean))
        spemean = np.concatenate([spemean, np.zeros(Length_pe - len(spemean))])
        l = len(ent)
        print('{} waveforms will be computed'.format(l))
        dt = np.zeros(l * (Length_pe//5), dtype=opdt)
        start = 0
        end = 0
        for i in range(l):
            wf_input = ent[i]['Waveform']
            wf_input = -1 * epulse * wf_input
            #wave = wf_input - wfaf.find_base_fast(wf_input)
            wave = wf_input - wfaf.find_base(wf_input, m_l, thres)
            pf = lucyDDM(wave, spemean, 50)

            if np.sum(pf <= 0.1) == len(pf):
                t = np.where(wave == wave.min())[0][:1] - np.argmin(spemean)
                possible = t if t[0] >= 0 else np.array([0])
                pf = np.array([1])
            pwe = pf[pf > 0.1]
            pwe = pwe.astype(np.float16)
            lenpf = len(pwe)
            pet = possible[pf > 0.1]
            end = start + lenpf
            dt['PETime'][start:end] = pet
            dt['Weight'][start:end] = pwe
            dt['EventID'][start:end] = ent[i]['EventID']
            dt['ChannelID'][start:end] = ent[i]['ChannelID']
            start = end
            print('\rAnsw Generating:|{}>{}|{:6.2f}%'.format(((20*i)//l)*'-', (19-(20*i)//l)*' ', 100 * ((i+1) / l)), end='' if i != l-1 else '\n')
    dt = dt[dt['Weight'] > 0]
    dt = np.sort(dt, kind='stable', order=['EventID', 'ChannelID', 'PETime'])
    with h5py.File(fopt, 'w') as opt:
        dset = opt.create_dataset('Answer', data=dt, compression='gzip')
        print('The output file path is {}'.format(fopt), end=' ', flush=True)
    return

if __name__ == '__main__':
    main(args.opt, args.ipt, args.ref)
