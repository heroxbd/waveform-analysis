# -*- coding: utf-8 -*-

import os
import math
import numpy as np
np.set_printoptions(suppress=True)
import scipy
import scipy.stats
from scipy import optimize as opti
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_axes_aligner import align
import numba
import h5py
from scipy.interpolate import interp1d

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 8
plt.rcParams['lines.markersize'] = 4.0
plt.rcParams['lines.linewidth'] = 1.0

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
        t = np.argwhere(wave == wave.max())[0][0] - spe_pre['peak_c']
        possible = np.array([t]) if t >= 0 else np.array([0])
        pf_r = np.array([1])
    pf = np.zeros_like(wave)
    pf[possible] = pf_r
    return pf, possible

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

def lucyddm_core(waveform, spe_pre, iterations=100):
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
    wave = wave / np.sum(spe_pre['spe'])
    t = np.argwhere(spe_pre['spe'] > 0)[0][0]
    spe_t = spe_pre['spe'][spe_pre['spe'] > 0]
    l = len(spe_t)
    # use the deconvlution method
    wave_deconv = np.full(wave.shape, 0.1)
    spe_mirror = spe_t[::-1]
    for _ in range(iterations):
        relative_blur = wave / np.convolve(wave_deconv, spe_t, 'same')
        wave_deconv = wave_deconv * np.convolve(relative_blur, spe_mirror, 'same')
        # there is no need to set the bound if the spe and the wave are all none negative
    wave_deconv = np.append(wave_deconv[(l-1)//2+t:], np.zeros((l-1)//2+t))
    # np.convolve(wave_deconv, spe, 'full')[:len(wave)] should be wave
    return wave_deconv

def threshold(wave, spe_pre):
    pet = np.argwhere(wave[spe_pre['peak_c']:] > spe_pre['thres']).flatten()
    pwe = wave[spe_pre['peak_c']:][pet] / np.sum(spe_pre['spe'])
    if len(pet) == 0:
        t = np.argwhere(wave == wave.max())[0][0] - spe_pre['peak_c']
        pet = np.array([t]) if t >= 0 else np.array([0])
        pwe = np.array([1])
    return pet, pwe

def norm_fit(x, M, y, eta=0):
    return np.power(y - np.matmul(M, x), 2).sum() + eta * x.sum()

def read_model(spe_path):
    with h5py.File(spe_path, 'r', libver='latest', swmr=True) as speFile:
        cid = speFile['SinglePE'].attrs['ChannelID']
        epulse = speFile['SinglePE'].attrs['Epulse']
        spe = speFile['SinglePE'].attrs['SpePositive']
        thres = speFile['SinglePE'].attrs['Thres']
        spe_pre = {}
        fig = plt.figure()
        fig.tight_layout()
        ax = fig.add_subplot(111)
        for i in range(len(spe)):
            m_l = np.sum(spe[i] > thres[i])
            peak_c = np.argmax(spe[i])
            mar_l = np.sum(spe[i][:peak_c] < thres[i])
            mar_r = np.sum(spe[i][peak_c:] < thres[i])
            spe_pre_i = {'spe':spe[i], 'epulse':epulse, 'peak_c':peak_c, 'm_l':m_l, 'mar_l':mar_l, 'mar_r':mar_r, 'thres':thres[i]}
            spe_pre.update({cid[i]:spe_pre_i})
            ax.plot(spe_pre[cid[i]]['spe'])
        ax.grid()
        ax.set_xlabel('$Time/\mathrm{ns}$')
        ax.set_ylabel('$Voltage/\mathrm{mV}$')
        fig.savefig('img/spe.png', bbox_inches='tight')
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

def demo(pet, pwe, tth, spe_pre, leng, possible, wave, cid, mode):
    print('possible = {}'.format(possible))
    penum = len(tth)
    print('PEnum is {}'.format(penum))
    pf0 = np.zeros(leng); pf1 = np.zeros(leng)
    if mode == 'Weight':
        tru_pet = tth['RiseTime']
        t, c = np.unique(tru_pet, return_counts=True)
        pf0[t] = c
        pf1[pet] = pwe
        xlabel = '$PEnum/\mathrm{1}$'
        distd = '(W/ns,P/1)'; distl = 'pdist'
        Q = penum; q = np.sum(pwe)
        edist = np.abs(Q - q) * scipy.stats.poisson.pmf(Q, Q)
    elif mode == 'Charge':
        t = tth['RiseTime']; w = tth[mode]
        t = np.unique(t); c = np.array([np.sum(w[t == i]) for i in t])
        pf0[t] = c / spe_pre['spe'].sum()
        pf1[pet] = pwe / spe_pre['spe'].sum()
        xlabel = '$Charge/\mathrm{mV}\cdot\mathrm{ns}$'
        distd = '(W/ns,C/mV*ns)'; distl = 'cdiff'
        edist = pwe.sum() - c.sum()
    print('truth RiseTime = {}, Weight = {}'.format(t, c))
    wave0 = np.convolve(spe_pre['spe'], pf0, 'full')[:leng]
    print('truth Resi-norm = {}'.format(np.linalg.norm(wave-wave0)))
    print('RiseTime = {}, Weight = {}'.format(pet, pwe))
    wdist = scipy.stats.wasserstein_distance(t, pet, u_weights=c, v_weights=pwe)
    print('wdist = {},'.format(wdist)+distl+' = {}'.format(edist))
    wave1 = np.convolve(spe_pre['spe'], pf1, 'full')[:leng]
    print('Resi-norm = {}'.format(np.linalg.norm(wave-wave1)))

    fig = plt.figure()
    fig.tight_layout()
    ax = fig.add_subplot(111)
    ax.grid()
    ax2 = ax.twinx()
    ax.plot(wave, c='b', label='origin wave')
    ax.plot(wave0, c='k', label='truth wave')
    ax.plot(wave1, c='C1', label='recon wave')
    ax.scatter(possible, wave[possible], marker='+', c='r', label='possible')
    ax.set_xlabel('$Time/\mathrm{ns}$')
    ax.set_ylabel('$Voltage/\mathrm{mV}$')
    ax.set_xlim(250, 500)
    ax.hlines(spe_pre['thres'], 0, 1029, color='c', label='threshold')
    ax2.set_ylabel(xlabel)
    fig.suptitle('eid={},cid={},'.format(tth['EventID'][0], tth['ChannelID'][0])+distd+'-dist={:.2f},{:.2f}'.format(wdist, edist))
    ax2.vlines(t, 0, c, color='g', label='truth '+mode)
    ax2.vlines(pet, -pwe, 0, color='y', label='recon '+mode)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    align.yaxes(ax, 0, ax2, 0)
    ax2.legend(lines + lines2, labels + labels2)
    fig.savefig('img/demoe{}c{}.png'.format(tth['EventID'][0], tth['ChannelID'][0]), bbox_inches='tight')
    fig.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(spe_pre['spe'], c='b')
    ax.grid()
    ax.set_xlabel('$Time/\mathrm{ns}$')
    ax.set_ylabel('$Voltage/\mathrm{mV}$')
    fig.savefig('img/spe{}.png'.format(cid), bbox_inches='tight')
    fig.clf()
    return

