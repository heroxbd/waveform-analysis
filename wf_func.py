# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import scipy
import scipy.stats
from scipy import optimize as opti
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import interp1d

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 8
plt.rcParams['lines.markersize'] = 4.0
plt.rcParams['lines.linewidth'] = 1.0

Eta = 1
Er = 0.01

def fit_N(wave, spe_pre, method, eta=0):
    l = wave.shape[0]
    spe_l = spe_pre['spe'].shape[0] 
    n = math.ceil(spe_l/10)
    difth = np.sort(spe_pre['spe'][np.arange(2,spe_l-1)+1]-spe_pre['spe'][np.arange(2,spe_l-1)]-spe_pre['spe'][np.arange(2,spe_l-1)-1]+spe_pre['spe'][np.arange(2,spe_l-1)-2])[n]
    lowp = np.argwhere(vali_base(wave, spe_pre['m_l'], spe_pre['thres']) == 1).flatten()
    lowp = lowp[np.logical_and(lowp > 1, lowp < l-1)]
    flag = 1
    if len(lowp) != 0:
        panel = np.zeros(l)
        for j in lowp:
            head = j-spe_pre['mar_l'] if j-spe_pre['mar_l'] > 0 else 0
            tail = j+spe_pre['mar_r'] if j+spe_pre['mar_r'] <= l else l
            panel[head : tail + 1] = 1
        fitp = np.argwhere(panel == 1).flatten()
        numb = np.argwhere(wave[lowp+1]-wave[lowp]-wave[lowp-1]+wave[lowp-2] < difth).flatten()
        if len(numb) != 0:
            ran = np.arange(spe_pre['peak_c'] - 1, spe_pre['peak_c'] + 2)
            possible = np.unique(lowp[numb] - ran.reshape(ran.shape[0], 1))
            possible = possible[np.logical_and(possible>=0, possible<l)]
            if len(possible) != 0:
                if method == 'xiaopeip':
                    pf_r = xiaopeip_core(wave, spe_pre['spe'], fitp, possible, eta=eta)
            else:
                flag = 0
        else:
            flag = 0
    else:
        flag = 0
    if flag == 0:
        t = np.where(wave == wave.min())[0][:1] - spe_pre['peak_c']
        possible = t if t[0] >= 0 else np.array([0])
        pf_r = np.array([1])
    pf = np.zeros_like(wave)
    pf[possible] = pf_r
    return pf

def xiaopeip_core(wave, spe, fitp, possible, eta=0):
    l = wave.shape[0]
    spe = np.concatenate([spe, np.zeros(l - spe.shape[0])])
    ans0 = np.zeros_like(possible).astype(np.float64)
    b = np.zeros((possible.shape[0], 2)).astype(np.float64)
    b[:, 1] = np.inf
    mne = spe[np.mod(fitp.reshape(fitp.shape[0], 1) - possible.reshape(1, possible.shape[0]), l)]
    ans = opti.fmin_l_bfgs_b(norm_fit, ans0, args=(mne, wave[fitp], eta), approx_grad=True, bounds=b, maxfun=500000)
    # ans = opti.fmin_slsqp(norm_fit, ans0, args=(mne, wave[fitp]), bounds=b, iprint=-1, iter=500000)
    # ans = opti.fmin_tnc(norm_fit, ans0, args=(mne, wave[fitp]), approx_grad=True, bounds=b, messages=0, maxfun=500000)
    pf_r = ans[0]
    return pf_r

def lucyddm_core(waveform, spe, iterations=100):
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
    wave = np.where(waveform > 0, waveform, 0)
    wave = wave + 0.001
    wave = wave / np.sum(spe)
    t = np.argwhere(spe > 0)[0][0]
    spe_t = spe[spe > 0]
    l = spe_t.shape[0]
    # use the deconvlution method
    wave_deconv = np.full(wave.shape, 0.1)
    spe_mirror = spe_t[::-1]
    for _ in range(iterations):
        relative_blur = wave / np.convolve(wave_deconv, spe_t, 'same')
        wave_deconv = wave_deconv * np.convolve(relative_blur, spe_mirror, 'same')
        # there is no need to set the bound if the spe and the wave are all none negative
    wave_deconv = np.append(wave_deconv[(l-1)//2+t:], np.zeros((l-1)//2+t))
    # np.convolve(wave_deconv, spe, 'full')[:len(wave)] should be wave
    wave_deconv = np.where(wave_deconv<50, wave_deconv, 0)
    return wave_deconv

def ergodic(wave, mne, base):
    n = mne.shape[1]; omega = 2**n
    b = np.argmin([norm_fit(np.array(list(['{:0'+str(n)+'b}'][0].format(i))).astype(np.float16) + base, mne, wave, eta=0) for i in range(omega)])
    return np.array(list(['{:0'+str(n)+'b}'][0].format(b))).astype(np.float16) + base

def xpp_convol(pet, wgt):
    core = np.array([0.9, 1.7, 0.9])
    idt = np.dtype([('PETime', np.int16), ('Weight', np.float16), ('Wgt_b', np.uint8)])
    seg = np.zeros(np.max(pet) + 3, dtype=idt)
    seg['PETime'] = np.arange(-1, np.max(pet) + 2)
    seg['Weight'][np.sort(pet) + 1] = wgt[np.argsort(pet)]
    seg['Wgt_b'] = np.around(seg['Weight'])
    resi = seg['Weight'][1:-1] - seg['Wgt_b'][1:-1]
    t = np.convolve(resi, core, 'full')
    ta = np.diff(t, prepend=t[0])
    tb = np.diff(t, append=t[-1])
    seg['Wgt_b'][(ta > 0)*(tb < 0)*(t > 0.5)*(seg['Wgt_b'] == 0.0)*(seg['Weight'] > 0)] += 1
    if np.sum(seg['Wgt_b'][1:-1] > 0) != 0:
        pwe = seg['Wgt_b'][1:-1][seg['Wgt_b'][1:-1] > 0]
        pet = seg['PETime'][1:-1][seg['Wgt_b'][1:-1] > 0]
    else:
        pwe = np.array([1])
        pet = seg['PETime'][np.argmax(seg['Weight'])]
    return pet, pwe

def potential(t, pet, resi):
    r0 = np.abs(t.reshape(len(t), 1) - pet.reshape(1, len(pet))) + Er
    p0 = np.exp(Eta * -r0) / r0 * resi
    r1 = np.abs(t.reshape(len(t), 1) - t.reshape(1, len(t))) + Er; np.fill_diagonal(r1, 1)
    p1 = np.exp(Eta * -r1) / r1
    return np.sum(p1) - np.sum(p0)

def hybird_select(pet, pwe, wave, spe, Thres):
    n = np.sum(pwe > Thres)
    if n == 1:
        pet_a = pet[pwe > Thres]
        pwe_a = np.array([1])
    elif n <= 1:
        pet_a = pet[pwe > Thres]
        l = len(wave); panel = np.arange(l)
        spe = np.append(spe, np.zeros(l - len(spe)))
        mne = spe[np.mod(panel.reshape(panel.shape[0], 1) - pet_a.reshape(1, pet_a.shape[0]), l)]
        pwe_a = ergodic(wave, mne, np.floor(pwe[pwe > Thres]))
    else:
        pet_a, pwe_a = xpp_convol(pet, pwe)
#         pwe_b = np.floor(pwe)
#         resi = pwe - pwe_b
#         pen = int(np.round(np.sum(resi)))
#         if pen > 0:
#             b = np.zeros((pen, 2)).astype(np.float64); b[:, 1] = np.inf
#             ans = opti.fmin_l_bfgs_b(potential, pet[np.argsort(resi)[-pen:]] + 0.3, args=(pet, resi), approx_grad=True, bounds=b)
#             t = ans[0]
#             pet_a = np.append(pet, t)
#         else:
#             pet_a = pet
#         pwe_a = np.append(pwe_b, np.ones(pen))
#         pwe_a = pwe_a[np.argsort(pet_a)]; pet_a = np.sort(pet_a)
    pet_a = pet_a[pwe_a > 0]; pwe_a = pwe_a[pwe_a > 0]
    if not len(pet_a) > 0:
        pet_a = pet[np.argmax(pwe)]; pwe_a = np.array([1])
    return pet_a, pwe_a

def norm_fit(x, M, y, eta=0):
    return np.power(y - np.matmul(M, x), 2).sum() + eta * x.sum()

def read_model(spe_path):
    with h5py.File(spe_path, 'r', libver='latest', swmr=True) as speFile:
        cid = speFile['SinglePE'].attrs['ChannelID']
        epulse = speFile['SinglePE'].attrs['Epulse']
        spe = speFile['SinglePE'].attrs['SpePositive']
        thres = speFile['SinglePE'].attrs['Thres']
        spe_pre = {}
        for i in range(len(spe)):
            m_l = np.sum(spe[i] > thres[i])
            peak_c = np.argmax(spe[i])
            mar_l = np.sum(spe[i][:peak_c] < thres[i])
            mar_r = np.sum(spe[i][peak_c:] < thres[i])
            spe_pre_i = {'spe':spe[i], 'epulse':epulse, 'peak_c':peak_c, 'm_l':m_l, 'mar_l':mar_l, 'mar_r':mar_r, 'thres':thres[i]}
            spe_pre.update({cid[i]:spe_pre_i})
            plt.plot(spe_pre[cid[i]]['spe'])
        plt.grid()
        plt.xlabel(r'Time/[ns]')
        plt.ylabel(r'ADC')
        plt.savefig('spe.png')
        plt.close()
    return spe_pre

def snip_baseline(waveform, itera=20):
    wm = np.min(waveform)
    waveform = waveform - wm
    v = np.log(np.log(np.sqrt(waveform+1)+1)+1)
    N = waveform.shape[0]
    for i in range(itera):
        v[i:N-i] = np.minimum(v[i:N-i], (v[:N-2*i] + v[2*i:])/2)
    w = np.power(np.exp(np.exp(v) - 1) - 1, 2) - 1 + wm
    return w

def vali_base(waveform, m_l, thres):
    m = np.median(waveform[waveform < np.median(waveform)])
    vali = np.where(waveform - m > thres, 1, 0) # valid waveform, not dark noise
    pos = omi2pos(vali)
    pos = rm_frag(pos, m_l)
    vali = pos2omi(pos, waveform.shape[0])
    return vali

def deduct_base(waveform, m_l=None, thres=None, itera=20, mode='fast'):
    def deduct(wave):
        wave = wave - np.min(wave)
        baseline = snip_baseline(wave, itera)
        wave = wave - baseline
        if mode == 'detail':
            wave = wave - find_base(wave, m_l, thres)
        elif mode == 'fast':
            wave = wave - find_base_fast(wave)
        return wave
    if waveform.ndim == 2:
        return np.array([deduct(waveform[i]) for i in range(len(waveform))])
    else:
        return deduct(waveform)

def find_base(waveform, m_l, thres):
    vali = vali_base(waveform, m_l, thres)
    base_line = np.mean(waveform[vali == 0])
    return base_line

def find_base_fast(waveform):
    m = np.median(waveform[waveform < np.median(waveform)])
    base_line = np.mean(waveform[np.logical_and(waveform < m + 4, waveform > m - 4)])
    return base_line

def omi2pos(vali):
    vali_t = np.concatenate((np.array([0]), vali, np.array([0])), axis=0)
    dval = np.diff(vali_t)
    pos_begin = np.argwhere(dval == 1).flatten()
    pos_end = np.argwhere(dval == -1).flatten()
    pos = np.concatenate((pos_begin.reshape(pos_begin.shape[0], 1), pos_end.reshape(pos_end.shape[0], 1)), axis = 1).astype(np.int16)
    return pos

def pos2omi(pos, len_n):
    vali = np.zeros(len_n).astype(np.int16)
    for i in range(pos.shape[0]):
        vali[pos[i][0]:pos[i][1]] = 1
    return vali

def rm_frag(pos, m_l):
    n = pos.shape[0]
    pos_t = []
    for i in range(n):
        if pos[i][1] - pos[i][0] > m_l//3:
            pos_t.append(pos[i])
    pos = np.array(pos_t)
    return pos

def pf_to_tw(pf, thres = 0):
    assert thres < 0.5, 'thres is too large, which is {}'.format(thres)
    if np.max(pf) < thres:
        t = np.argmax(pf)
        pf = np.zeros_like(pf)
        pf[t] = 1
    pwe = pf[pf > thres]
    pet = np.argwhere(pf > thres).flatten()
    return pet, pwe
1
def showwave(pet, pwe, spe, leng):
    print(spe.shape)
    spe = np.concatenate((np.zeros(leng), spe)); spe = np.concatenate((spe, np.zeros(leng)))
    time = np.arange(-leng, -leng + len(spe))
    f = interp1d(time, spe)
    wave = np.zeros(leng)
    tp = np.arange(leng)
    for i in range(len(pet)):
        wave += f(tp - pet[i]) * pwe[i]
    return wave

def demo(pet, pwe, tth, spe_pre, leng, possible, wave, cid):
    print('possible = {}'.format(possible))
    penum = len(tth)
    print('PEnum is {}'.format(penum))
    tru_pet = tth['PETime']
    pet0, pwe0 = np.unique(tru_pet, return_counts=True)
    print('truth PETime = {}, Weight = {}'.format(pet0, pwe0))
    pf0 = np.zeros(leng); pf0[pet0] = pwe0
    wave0 = np.convolve(spe_pre['spe'], pf0, 'full')[:leng]
    print('truth Resi-norm = {}'.format(np.linalg.norm(wave-wave0)))
    print('before PETime = {}, Weight = {}'.format(pet, pwe))
    wdist = scipy.stats.wasserstein_distance(tru_pet, pet, v_weights=pwe)
    Q = penum; q = np.sum(pwe)
    pdist = np.abs(Q - q) * scipy.stats.poisson.pmf(Q, Q)
    print('before wdist is {}, pdist is {}'.format(wdist, pdist))
    pf1 = np.zeros(leng); pf1[pet] = pwe
    wave1 = np.convolve(spe_pre['spe'], pf1, 'full')[:leng]
    print('before Resi-norm = {}'.format(np.linalg.norm(wave-wave1)))
    pet_a, pwe_a = hybird_select(pet, pwe, wave, spe_pre['spe'], 0.2)
    print('after PETime = {}, Weight = {}'.format(pet_a, pwe_a))
    wdist_a = scipy.stats.wasserstein_distance(tru_pet, pet_a, v_weights=pwe_a)
    q_a = np.sum(pwe_a)
    pdist_a = np.abs(Q - q_a) * scipy.stats.poisson.pmf(Q, Q)
    print('after wdist is {}, pdist is {}'.format(wdist_a, pdist_a))
    wave_a = showwave(pet_a, pwe_a, spe_pre['spe'], leng)
    print('after Resi-norm = {}'.format(np.linalg.norm(wave-wave_a)))
    plt.plot(wave, c='b', label='original WF')
    plt.plot(wave0, c='k', label='truth WF')
    plt.plot(wave1, c='y', label='before adjust WF')
    plt.plot(wave_a, c='m', label='after adjust WF')
    plt.scatter(possible, wave[possible], marker='+', c='r')
    plt.xlabel(r'Time/[ns]')
    plt.ylabel(r'ADC')
    plt.xlim(200, 400)
    plt.hlines(spe_pre['thres'], 200, 500, color='c')
    t, c = np.unique(tru_pet, return_counts=True)
    plt.vlines(t, 0, 10*c, color='k')
    hh = -10*(np.max(pwe_a)+1)
    plt.vlines(pet, -10*pwe+hh, hh, color='y')
    plt.vlines(pet_a, -10*pwe_a, 0, color='m')
    plt.legend()
    plt.grid()
    plt.title('eid={}, cid={}, wp-dist=({:.2f},{:.2f})->({:.2f},{:.2f})'.format(tth['EventID'][0], tth['ChannelID'][0], wdist, pdist, wdist_a, pdist_a))
    plt.savefig('demoe{}c{}.png'.format(tth['EventID'][0], tth['ChannelID'][0]))
    plt.close()
    plt.plot(spe_pre['spe'], c='b')
    plt.grid()
    plt.xlabel(r'Time/[ns]')
    plt.ylabel(r'ADC')
    plt.savefig('spe{}.png'.format(cid))
    plt.close()
    return

