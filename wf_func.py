import os
import math
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

import numpy as np
np.set_printoptions(suppress=True)
import scipy
import scipy.stats
from scipy.stats import poisson, uniform, norm
from scipy.fftpack import fft, ifft
from scipy import optimize as opti
import scipy.special as special
from scipy.signal import convolve
from scipy.signal import savgol_filter
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_axes_aligner import align
import h5py
from scipy.interpolate import interp1d
import numba

matplotlib.use('pgf')
plt.style.use('classic')
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 20
plt.rcParams['lines.markersize'] = 4.0
plt.rcParams['lines.linewidth'] = 2.0
# plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['text.usetex'] = True
plt.rcParams['pgf.texsystem'] = 'pdflatex'
plt.rcParams['axes.unicode_minus'] = False

def xiaopeip(wave, spe_pre, eta=0):
    l = len(wave)
    flag = 1
    lowp = np.argwhere(wave > 5 * spe_pre['std']).flatten()
    if len(lowp) != 0:
        fitp = np.arange(lowp.min() - round(spe_pre['mar_l']), lowp.max() + round(spe_pre['mar_r']))
        fitp = np.unique(np.clip(fitp, 0, len(wave)-1))
        pet = lowp - spe_pre['peak_c']
        pet = np.unique(np.clip(pet, 0, len(wave)-1))
        if len(pet) != 0:
#             cha, ped = xiaopeip_core(wave, spe_pre['spe'], fitp, pet, eta=eta)
            cha = xiaopeip_core(wave, spe_pre['spe'], fitp, pet, eta=eta)
            cha = cha / np.sum(cha) * np.sum(wave) / np.sum(spe_pre['spe'])
        else:
            flag = 0
    else:
        flag = 0
    if flag == 0:
        pet = np.array([np.argmax(wave[spe_pre['peak_c']:])])
        cha = np.array([1])
#     return pet, cha, ped
    return pet, cha

# def xiaopeip_core(wave, spe, fitp, possible, eta=0):
#     l = len(wave)
#     spe = np.concatenate([spe, np.zeros(l - len(spe))])
#     ans0 = np.zeros(len(possible)+1).astype(np.float64)
#     ans0[-1] = wave.min()
#     b = np.zeros((len(possible)+1, 2)).astype(np.float64)
#     b[-1, 0] = -np.inf
#     b[:, 1] = np.inf
#     mne = spe[np.mod(fitp.reshape(len(fitp), 1) - possible.reshape(1, len(possible)), l)]
#     ans = opti.fmin_l_bfgs_b(norm_fit, ans0, args=(mne, wave[fitp], eta), approx_grad=True, bounds=b, maxfun=500000)
#     # ans = opti.fmin_slsqp(norm_fit, ans0, args=(mne, wave[fitp]), bounds=b, iprint=-1, iter=500000)
#     # ans = opti.fmin_tnc(norm_fit, ans0, args=(mne, wave[fitp]), approx_grad=True, bounds=b, messages=0, maxfun=500000)
#     pf = ans[0]
#     return pf[:-1], pf[-1]

# def norm_fit(x, M, y, eta=0):
#     return np.power(y - x[-1] - np.matmul(M, x[:-1]), 2).sum() + eta * x.sum()

def xiaopeip_core(wave, spe, fitp, possible, eta=0):
    l = len(wave)
    spe = np.concatenate([spe, np.zeros(l - len(spe))])
    ans0 = np.ones(len(possible)).astype(np.float64)
    b = np.zeros((len(possible), 2)).astype(np.float64)
    b[:, 1] = np.inf
    mne = spe[np.mod(fitp.reshape(len(fitp), 1) - possible.reshape(1, len(possible)), l)]
    try:
        ans = opti.fmin_l_bfgs_b(norm_fit, ans0, args=(mne, wave[fitp], eta), approx_grad=True, bounds=b, maxfun=500000)
    except ValueError:
        ans = [np.ones(len(possible)) * 0.2]
    # ans = opti.fmin_slsqp(norm_fit, ans0, args=(mne, wave[fitp]), bounds=b, iprint=-1, iter=500000)
    # ans = opti.fmin_tnc(norm_fit, ans0, args=(mne, wave[fitp]), approx_grad=True, bounds=b, messages=0, maxfun=500000)
    return ans[0]

def norm_fit(x, M, y, eta=0):
    return np.power(y - np.matmul(M, x), 2).sum() + eta * x.sum()

def lucyddm(waveform, spe_pre):
    '''Lucy deconvolution
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
    .. [2] https://github.com/scikit-image/scikit-image/blob/master/skimage/restoration/deconvolution.py#L329
    '''
    spe = np.append(np.zeros(len(spe_pre['spe']) - 2 * 9 - 1), np.abs(spe_pre['spe']))
    waveform = np.where(waveform <= 0, 1e-6, waveform)
    spe = np.where(spe <= 0, 1e-6, spe)
    waveform = waveform / np.sum(spe)
    wave_deconv = waveform.copy()
    spe_mirror = spe[::-1]
    while True:
        relative_blur = waveform / np.convolve(wave_deconv, spe, mode='same')
        new_wave_deconv = wave_deconv * np.convolve(relative_blur, spe_mirror, mode='same')
        if np.max(np.abs(wave_deconv[9:] - new_wave_deconv[9:])) < 1e-4:
            break
        else:
            wave_deconv = new_wave_deconv
    return np.arange(0, len(waveform) - 9), wave_deconv[9:]

def waveformfft(wave, spe_pre):
    w = savgol_filter(wave, 11, 2)
    lowp = np.argwhere(w > 5 * spe_pre['std']).flatten()
    if len(lowp) != 0:
        left = np.clip(lowp.min() - round(2 * spe_pre['mar_l']), 0, len(wave) - 1)
        right = np.clip(lowp.max() + round(2 * spe_pre['mar_r']), 0, len(wave) - 1)
        pet = np.arange(left, right)
        w = w[left:right]
        length = len(w)
        spefft = fft(spe_pre['spe'], 2*length)
        wavef = fft(w, 2*length)
        wavef[(length-round(length*0.8)):(length+round(length*0.8))] = 0
        signalf = np.true_divide(wavef, spefft)
        recon = np.real(ifft(signalf, 2*length))
        cha = recon[:length]
        cha = np.abs(cha / np.sum(cha) * np.abs(np.sum(wave)) / np.sum(spe_pre['spe']))
    else:
        pet = np.argmax(wave[spe_pre['peak_c']:]).flatten()
        cha = np.array([1])
    return pet, cha

def threshold(wave, spe_pre):
    pet = np.argwhere(wave[spe_pre['peak_c']:] > spe_pre['std'] * 5).flatten()
    cha = wave[spe_pre['peak_c']:][pet]
    cha = cha / cha.sum() * np.abs(wave.sum()) / spe_pre['spe'].sum()
    if len(pet) == 0:
        pet = np.array([np.argmax(wave[spe_pre['peak_c']:])])
        cha = np.array([1])
    return pet, cha

def findpeak(wave, spe_pre):
    w = savgol_filter(wave, 11, 2)
    dpta = np.where(np.diff(w, prepend=w[0]) > 0, 1, -1)
    dpta = np.diff(dpta, prepend=dpta[0])
    petr = np.argwhere((w > spe_pre['std'] * 5) & (dpta < 0)).flatten() - spe_pre['peak_c']
    pet = petr[petr >= 0]
    cha = wave[pet + spe_pre['peak_c']]
    cha = cha / np.sum(cha) * np.abs(np.sum(wave)) / np.sum(spe_pre['spe'])
    if len(pet) == 0:
        pet = np.array([np.argmax(wave[spe_pre['peak_c']:])])
        cha = np.array([1])
    return pet, cha

def fbmp(y, spe_pre, tlist, A, p1, sig2s, mus, D, stop=0):
    xmmse = fbmpr_fxn_reduced(y, A, p1, spe_pre['std'] ** 2, sig2s, mus, D, stop=0)[0]
    pet = tlist[xmmse > 0]
    cha = xmmse[xmmse > 0] / mus[1]
    if len(pet) == 0:
        pet = np.array([np.argmax(y[spe_pre['peak_c']:])])
        cha = np.array([1])
    return pet, cha

def fbmpr_fxn_reduced(y, A, p1, sig2w, sig2s, mus, D, stop=0):
    M, N = A.shape
    Q = len(mus) - 1
    ps = np.insert(np.ones(Q) * p1 / Q, 0, 1 - p1)
    sig2s = np.hstack([sig2s, sig2s[1] * np.ones(Q - 1)])

    a2 = np.trace(np.matmul(A.T, A)) / N
    nu_true_mean = - M / 2 - M / 2 * np.log(sig2w) - p1 * N / 2 * np.log(a2 * sig2s[1] / sig2w + 1) - M / 2 * np.log(2 * np.pi) + N * np.log(ps[0]) + p1 * N * np.log(ps[1] / ps[0])
    nu_true_stdv = np.sqrt(M / 2 + N * p1 * (1 - p1) * (np.log(ps[1] / ps[0]) - np.log(a2 * sig2s[1] / sig2w + 1) / 2) ** 2)
    nu_stop = nu_true_mean - stop * nu_true_stdv

    psy_thresh = 1e-4
    P = int(min(M, 1 + np.ceil(N * p1 + special.erfcinv(1e-2) * np.sqrt(2 * N * p1 * (1 - p1)))))

    T = [[np.empty(0).astype(int) for i in range(D)] for i in range(P)]
    sT = [[np.empty(0).astype(int) for i in range(D)] for i in range(P)]
    nu = -np.inf * np.ones([P, D])
    xmmse = [[np.empty(0).astype(int) for i in range(D)] for i in range(P)]
    d_tot = np.inf

    nu_root = -np.linalg.norm(y) ** 2 / 2 / sig2w - M * np.log(2 * np.pi) / 2 - M * np.log(sig2w) / 2 + N * np.log(ps[0])
    Bxt_root = A / sig2w
    betaxt_root = abs(sig2s[1] * (1 + sig2s[1] * sum(np.conjugate(A) * Bxt_root)) ** (-1))
    nuxt_root = np.zeros(Q * N)
    for q in range(Q):
        nuxt_root[q * N: N + q * N] = nu_root + np.log(betaxt_root / sig2s[1]) / 2 + 0.5 * betaxt_root * abs(np.matmul(y[None, :], Bxt_root) + mus[q + 1] / sig2s[1]) ** 2 - 0.5 * abs(mus[q + 1]) ** 2 / sig2s[1] + np.log(ps[1] / ps[0])
    
    for d in range(D):
        nuxt = np.empty(nuxt_root.shape)
        nuxt[:] = nuxt_root[:]
        z = np.empty(y.shape)
        z[:] = y[:]
        Bxt = np.empty(Bxt_root.shape)
        Bxt[:] = Bxt_root[:]
        betaxt = np.empty(betaxt_root.shape)
        betaxt[:] = betaxt_root[:]
        for p in range(P):
            nustar = max(nuxt)
            nqstar = np.argmax(nuxt)
            while sum(abs(nustar - nu[p, :d]) < 1e-8):
                nuxt[nqstar] = -np.inf
                nustar = max(nuxt)
                nqstar = np.argmax(nuxt)
            qstar = int(nqstar // N)
            nstar = int(nqstar % N)
            nu[p, d] = nustar
            T[p][d] = np.append(T[p - 1][d], nstar)
            sT[p][d] = np.append(sT[p - 1][d], qstar + 1)
            z = z - A[:, nstar] * mus[qstar + 1]
            Bxt = Bxt - np.matmul(Bxt[:, nstar][:, None] * betaxt[nstar], np.matmul(Bxt[:, nstar][None, :], A))
            xmmse[p][d] = np.zeros(N)
            xmmse[p][d][T[p][d]] = mus[sT[p][d]] + sig2s[1] * np.matmul(Bxt[:, T[p][d]].T, z[:, None]).flatten()
            betaxt = abs(sig2s[1] * (1 + sig2s[1] * np.sum(np.conjugate(A) * Bxt, axis=0)) ** (-1))
            for q in range(Q):
                nuxt[q * N: q * N + N] = nu[p][d] + np.log(betaxt / sig2s[1]) / 2 + 0.5 * betaxt * abs(np.matmul(z[None, :], Bxt) + mus[q + 1] / sig2s[1]) ** 2 - 0.5 * abs(mus[q + 1]) ** 2 / sig2s[1] + np.log(ps[1] / ps[0])
                nuxt[T[p][d] + q * N] = -np.inf * np.ones(T[p][d].shape)

        if max(nu[:, d]) > nu_stop:
            d_tot = d
            break
    nu = nu[:, :d+1]

    dum = np.sort(nu.T.flatten())[::-1]
    indx = np.argsort(nu.T.flatten())[::-1]
    d_max = int(np.floor(indx[0] / P))
    nu_max = nu.T.flatten()[indx[0]]
    num = sum(nu.flatten() > nu_max + np.log(psy_thresh))
    nu_star = nu.T.flatten()[indx[:num]]
    psy_star = np.exp(nu_star - nu_max) / sum(np.exp(nu_star - nu_max))
    T_star = [None for i in range(num)]
    xmmse_star = [None for i in range(num)]
    p1_up = 0
    for k in range(num):
        T_star[k] = T[indx[k] % P][indx[k] // P]
        xmmse_star[k] = xmmse[indx[k] % P][indx[k] // P]
        p1_up = p1_up + psy_star[k] * len(T_star[k]) / N

    xmmse = np.matmul(np.vstack([xmmse_star]).T, psy_star[:, None]).flatten()

    return xmmse, xmmse_star, psy_star, nu_star, T_star, d_tot, d_max

def read_model(spe_path):
    with h5py.File(spe_path, 'r', libver='latest', swmr=True) as speFile:
        cid = speFile['SinglePE'].attrs['ChannelID']
        epulse = speFile['SinglePE'].attrs['Epulse']
        spe = speFile['SinglePE'].attrs['SpePositive']
        std = speFile['SinglePE'].attrs['Std']
        if 'parameters' in list(speFile['SinglePE'].attrs.keys()):
            p = speFile['SinglePE'].attrs['parameters']
        else:
            p = [None,] * len(spe)
        spe_pre = {}
        fig = plt.figure()
        # fig.tight_layout()
        gs = gridspec.GridSpec(1, 1, figure=fig, left=0.1, right=0.85, top=0.95, bottom=0.15, wspace=0.4, hspace=0.5)
        ax = fig.add_subplot(gs[0, 0])
        for i in range(len(spe)):
            peak_c = np.argmax(spe[i])
            ft = interp1d(np.arange(0, len(spe[i]) - peak_c), 0.1 - spe[i][peak_c:])
            t = opti.fsolve(ft, x0=np.argwhere(spe[i][peak_c:] < 0.1).flatten()[0])[0] + peak_c
            fl = interp1d(np.arange(0, peak_c), spe[i][:peak_c] - 5 * std[i])
            mar_l = opti.fsolve(fl, x0=np.sum(spe[i][:peak_c] < 5 * std[i]))[0]
            fr = interp1d(np.arange(0, len(spe[i]) - peak_c), 5 * std[i] - spe[i][peak_c:])
            mar_r = t - (opti.fsolve(fr, x0=np.sum(spe[i][peak_c:] > 5 * std[i]))[0] + peak_c)
            spe_pre_i = {'spe':spe[i], 'epulse':epulse, 'peak_c':peak_c, 'mar_l':mar_l, 'mar_r':mar_r, 'std':std[i], 'parameters':p[i]}
            spe_pre.update({cid[i]:spe_pre_i})
            ax.plot(spe_pre[cid[i]]['spe'])
        ax.grid()
        ax.set_xlabel(r'$Time/\mathrm{ns}$')
        ax.set_ylabel(r'$Voltage/\mathrm{mV}$')
        fig.savefig('Note/figures/pmtspe.pgf')
        fig.savefig('Note/figures/pmtspe.pdf')
        plt.close()
    return spe_pre

def clip(pet, cha, thres):
    if len(pet[cha > thres]) == 0:
        pet = np.array([pet[np.argmax(cha)]])
        cha = np.array([1])
    else:
        pet = pet[cha > thres]
        cha = cha[cha > thres]
    return pet, cha

def glow(n, tau):
    return np.random.exponential(tau, size=n)

def transit(n, sigma):
    return np.random.normal(0, sigma, size=n)

def time(n, tau, sigma):
    if tau == 0:
        return np.sort(transit(n, sigma))
    elif sigma == 0:
        return np.sort(glow(n, tau))
    else:
        return np.sort(glow(n, tau) + transit(n, sigma))

def convolve_exp_norm(x, tau, sigma):
    if tau == 0.:
        y = norm.pdf(x, loc=0, scale=sigma)
    elif sigma == 0.:
        y = np.where(x >= 0., 1/tau * np.exp(-x/tau), 0.)
    else:
        alpha = 1/tau
        co = alpha/2. * np.exp(alpha*alpha*sigma*sigma/2.)
        x_erf = (alpha*sigma*sigma - x)/(np.sqrt(2.)*sigma)
        y = co * (1. - special.erf(x_erf)) * np.exp(-alpha*x)
    return y

def spe(t, tau, sigma, A):
    s = np.zeros_like(t).astype(np.float64)
    t0 = t[t > np.finfo(np.float64).tiny]
    s[t > np.finfo(np.float64).tiny] = A * np.exp(-1 / 2 * (np.log(t0 / tau) * np.log(t0 / tau) / sigma / sigma))
    return s

def charge(n, gmu, gsigma, thres=0):
    chargesam = norm.ppf(1 - uniform.rvs(scale=1-norm.cdf(thres, loc=gmu, scale=gsigma), size=n), loc=gmu, scale=gsigma)
    return chargesam

def probcharhitt(t0, hitt, probcharge, Tau, Sigma, npe):
    prob = np.where(npe > 0, probcharge * np.power(convolve_exp_norm(hitt - t0, Tau, Sigma), npe), 0)
    prob = np.sum(prob, axis=1) / np.sum(probcharge, axis=1)
    return prob

def likelihoodt0(hitt, char, gmu, gsigma, Tau, Sigma, npe, mode='charge'):
    b = [0., 600.]
    tlist = np.arange(b[0], b[1] + 1e-6, 0.2)
    if mode == 'charge':
        l = len(hitt)
        char = np.tile(char, (2 * npe + 1, 1)).T
        hitt = np.tile(hitt, (2 * npe + 1, 1)).T
        npe = np.round(char / gmu) + np.tile(np.arange(-npe, npe + 1), (l, 1))
        npe[npe <= 0] = np.nan
        probcharge = npeprobcharge(char, npe, gmu=gmu, gsigma=gsigma)
        probcharhitt(500, hitt, probcharge, Tau, Sigma, npe)
        logL = lambda t0 : -1 * np.sum(np.log(np.clip(probcharhitt(t0, hitt, probcharge, Tau, Sigma, npe), np.finfo(np.float64).tiny, np.inf)))
        # logL = lambda t0 : -1 * np.sum(np.log(np.clip(convolve_exp_norm(hitt - t0, Tau, Sigma) * (char / gmu), np.finfo(np.float64).tiny, np.inf)))
    elif mode == 'all':
        logL = lambda t0 : -1 * np.sum(np.log(np.clip(convolve_exp_norm(hitt - t0, Tau, Sigma), np.finfo(np.float64).tiny, np.inf)))
    logLv = np.vectorize(logL)
    t0 = opti.fmin_l_bfgs_b(logL, x0=[tlist[np.argmin(logLv(tlist))]], approx_grad=True, bounds=[b], maxfun=500000)[0]
    return t0

def npeprobcharge(charge, npe, gmu, gsigma):
    prob = np.where(npe > 0, norm.pdf(charge, loc=gmu * npe, scale=gsigma * np.sqrt(npe)) / (1 - norm.cdf(0, loc=gmu * npe, scale=gsigma * np.sqrt(npe))), 0)
    return prob

def stdrmoutlier(array, r):
    arrayrmoutlier = array[np.abs(array - np.mean(array)) < r * np.std(array, ddof=-1)]
    std = np.std(arrayrmoutlier, ddof=-1)
    return std, len(arrayrmoutlier)

def demo(pet_sub, cha_sub, tth, spe_pre, window, wave, cid, p, full=False, fold='Note/figures', ext='.pgf'):
    penum = len(tth)
    print('PEnum is {}'.format(penum))
    pan = np.arange(window)
    pet_ans_0 = tth['HitPosInWindow']
    cha_ans = tth['Charge']
    pet_ans = np.unique(pet_ans_0)
    cha_ans = np.array([np.sum(cha_ans[pet_ans_0 == i]) for i in pet_ans])
    ylabel = '$Charge/\mathrm{mV}\cdot\mathrm{ns}$'
    distd = '(W/ns,C/mV*ns)'
    distl = 'cdiff'
    edist = cha_sub.sum() - cha_ans.sum()
    print('truth HitPosInWindow = {}, Weight = {}'.format(pet_ans, cha_ans))
    wav_ans = np.sum([np.where(pan > pet_ans[j], spe(pan - pet_ans[j], tau=p[0], sigma=p[1], A=p[2]) * cha_ans[j] / spe_pre['spe'].sum(), 0) for j in range(len(pet_ans))], axis=0)
    print('truth Resi-norm = {}'.format(np.linalg.norm(wave - wav_ans)))
    print('HitPosInWindow = {}, Weight = {}'.format(pet_sub, cha_sub))
    wdist = scipy.stats.wasserstein_distance(pet_ans, pet_sub, u_weights=cha_ans, v_weights=cha_sub)
    print('wdist = {}, '.format(wdist) + distl + ' = {}'.format(edist))
    wav_sub = np.sum([np.where(pan > pet_sub[j], spe(pan - pet_sub[j], tau=p[0], sigma=p[1], A=p[2]) * cha_sub[j] / spe_pre['spe'].sum(), 0) for j in range(len(pet_sub))], axis=0)
    print('Resi-norm = {}'.format(np.linalg.norm(wave - wav_sub)))

    fig = plt.figure(figsize=(10, 10))
    fig.tight_layout()
    ax0 = fig.add_axes((.1, .2, .85, .3))
    ax0.plot(wave, c='b', label='origin wave')
    ax0.plot(wav_ans, c='k', label='truth wave')
    ax0.plot(wav_sub, c='g', label='recon wave')
    ax0.set_ylabel('$Voltage/\mathrm{mV}$')
    ax0.hlines(5 * spe_pre['std'], 0, len(wave), color='c', label='threshold')
    ax0.set_xticklabels([])
    ax0.set_ylim(min(wave)-5, max(wave)+5)
    ax0.legend(loc=1)
    ax0.grid()
    if full:
        ax0.set_xlim(0, window)
    else:
        ax0.set_xlim(max(pet_ans.min()-50, 0), min(pet_ans.max()+150, window))
    ax1 = fig.add_axes((.1, .5, .85, .2))
    ax1.vlines(pet_ans, 0, cha_ans, color='k', label='truth Charge')
    ax1.set_ylabel(ylabel)
    ax1.set_xticklabels([])
    ax1.set_xlim(ax0.get_xlim())
    ax1.set_ylim(0, max(max(cha_ans), max(cha_sub))*1.1)
    ax1.set_yticks(np.arange(0, max(max(cha_ans), max(cha_sub)), 50))
    ax1.legend(loc=1)
    ax1.grid()
    ax2 = fig.add_axes((.1, .7, .85, .2))
    ax2.vlines(pet_sub, 0, cha_sub, color='g', label='recon Charge')
    ax2.set_ylabel(ylabel)
    ax2.set_xticklabels([])
    ax2.set_xlim(ax0.get_xlim())
    ax2.set_ylim(0, max(max(cha_ans), max(cha_sub))*1.1)
    ax2.set_yticks(np.arange(0, max(max(cha_ans), max(cha_sub)), 50))
    ax2.legend(loc=1)
    ax2.grid()
    ax3 = fig.add_axes((.1, .1, .85, .1))
    ax3.scatter(np.arange(window), wav_sub - wave, c='k', label='residual wave', marker='.')
    ax3.set_xlabel('$t/\mathrm{ns}$')
    ax3.set_ylabel('$Voltage/\mathrm{mV}$')
    ax3.set_xlim(ax0.get_xlim())
    dh = int((max(np.abs(wav_sub - wave))//5+1)*5)
    ax3.set_yticks(np.linspace(-dh, dh, int(2*dh//5+1)))
    ax3.legend(loc=1)
    ax3.grid()
    if ext != '.pgf':
        fig.suptitle('eid={},cid={},'.format(tth['TriggerNo'][0], tth['PMTId'][0])+distd+'-dist={:.02f},{:.02f}'.format(wdist, edist), y=0.95)
    fig.savefig(fold + '/demoe{}c{}'.format(tth['TriggerNo'][0], tth['PMTId'][0]) + ext)
    fig.savefig(fold + '/demoe{}c{}'.format(tth['TriggerNo'][0], tth['PMTId'][0]) + '.pdf')
    fig.clf()
    plt.close(fig)

    fig = plt.figure()
    # fig.tight_layout()
    gs = gridspec.GridSpec(1, 1, figure=fig, left=0.1, right=0.85, top=0.95, bottom=0.15, wspace=0.4, hspace=0.5)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(spe_pre['spe'], c='b')
    ax.grid()
    ax.set_xlabel('$t/\mathrm{ns}$')
    ax.set_ylabel('$Voltage/\mathrm{mV}$')
    fig.savefig(fold + '/spe{:02d}'.format(cid) + ext)
    fig.savefig(fold + '/spe{:02d}'.format(cid) + '.pdf')
    fig.clf()
    plt.close(fig)