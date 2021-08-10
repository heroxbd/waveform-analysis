from operator import le
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
from matplotlib import cm
from matplotlib import colors
from mpl_axes_aligner import align
import h5py
from scipy.interpolate import interp1d
from sklearn.linear_model import orthogonal_mp
import warnings
warnings.filterwarnings('ignore')

matplotlib.use('pgf')
plt.style.use('default')
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 20
plt.rcParams['lines.markersize'] = 4.0
plt.rcParams['lines.linewidth'] = 2.0
# plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['text.usetex'] = True
plt.rcParams['pgf.texsystem'] = 'pdflatex'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pgf.preamble'] = r'\usepackage[detect-all,locale=DE]{siunitx}'

nshannon = 1
window = 1029
gmu = 160.
gsigma = 40.

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
    spe = np.append(np.zeros(len(spe_pre) - 1), np.abs(spe_pre))
    waveform = np.clip(waveform, 1e-6, np.inf)
    spe = np.clip(spe, 1e-6, np.inf)
    waveform = waveform / gmu
    wave_deconv = waveform.copy()
    spe_mirror = spe[::-1]
    while True:
        relative_blur = waveform / np.convolve(wave_deconv, spe, mode='same')
        new_wave_deconv = wave_deconv * np.convolve(relative_blur, spe_mirror, mode='same')
        if np.max(np.abs(wave_deconv - new_wave_deconv)) < 1e-4:
            break
        else:
            wave_deconv = new_wave_deconv
    return np.arange(len(waveform)), wave_deconv

def omp(wave, A, tlist, factor):
    coef = orthogonal_mp(A, wave[:, None])
    return tlist, coef * factor

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

def fbmpr_fxn_reduced(y, A, sig2w, sig2s, mus, D, p1, stop=0, truth=None, i=None, left=None, right=None, tlist=None, gmu=None, para=None, prior=True, plot=False):
    '''
    p1: prior probability for each bin.
    sig2w: variance of white noise.
    sig2s: variance of signal x_i.
    mus: mean of signal x_i.
    '''
    # Only for multi-gaussian with arithmetic sequence of mu and sigma
    M, N = A.shape

    psy_thresh = 0.1
    # upper limit of number of PEs.
    P = max(math.ceil(min(M, p1.sum() + 3 * np.sqrt(p1.sum()))), 1)

    T = np.full((P, D), 0)
    nu = np.full((P, D), -np.inf)
    xmmse = np.zeros((P, D, N))
    cc = np.zeros((P, D, N))

    # nu_root: nu for all s_n=0.
    nu_root = -0.5 * np.linalg.norm(y) ** 2 / sig2w - 0.5 * M * np.log(2 * np.pi) - 0.5 * M * np.log(sig2w)
    # nu_root = -0.5 * np.linalg.norm(y) ** 2 / sig2w - 0.5 * M * np.log(2 * np.pi)  # no Gaussian space factor
    if prior:
        nu_root = nu_root + poisson.logpmf(0, p1).sum()
    # Eq. (29)
    cx_root = A / sig2w
    # Eq. (30) sig2s = 1 sigma^2 - 0 sigma^2
    betaxt_root = sig2s / (1 + sig2s * np.einsum('ij,ij->j', A, cx_root, optimize=True))
    # Eq. (31)
    nuxt_root = nu_root + 0.5 * (betaxt_root * (y @ cx_root + mus / sig2s) ** 2  - mus ** 2 / sig2s) + 0.5 * np.log(betaxt_root / sig2s)
    # nuxt_root = nu_root + 0.5 * (betaxt_root * (y @ cx_root + mus / sig2s) ** 2  - mus ** 2 / sig2s)  # no Gaussian space factor
    if prior:
        nuxt_root = nuxt_root + poisson.logpmf(1, p1) - poisson.logpmf(0, p1)
    pan_root = np.zeros(N)

    # Repeated Greedy Search
    for d in range(D):
        nuxt = nuxt_root.copy()
        z = y.copy()
        cx = cx_root.copy()
        betaxt = betaxt_root.copy()
        pan = pan_root.copy()
        for p in range(P):
            # look for duplicates of nu and nuxt, set to -inf.
            # only inspect the same number of PEs in Row p.
            nuxtshadow = np.where(np.sum(np.abs(nuxt - nu[p:(p+1), :d].T) < 1e-4, axis=0), -np.inf, nuxt)
            nustar = max(nuxtshadow)
            istar = np.argmax(nuxtshadow)
            nu[p, d] = nustar
            T[p, d] = istar
            pan[istar] += 1
            # Eq. (33)
            cx -= np.einsum('n,m,mp->np', betaxt[istar] * cx[:, istar], cx[:, istar], A, optimize=True)

            # Eq. (34)
            z -= A[:, istar] * mus[istar]
            assist = np.zeros(N)
            t, c = np.unique(T[:p+1, d], return_counts=True)
            assist[t] = mus[t] * c + sig2s[t] * c * np.dot(z, cx[:, t])
            cc[p, d][t] = c
            xmmse[p, d] = assist

            # Eq. (30)
            betaxt = sig2s / (1 + sig2s * np.sum(A * cx, axis=0))
            # Eq. (31)
            nuxt = nustar + 0.5 * (betaxt * (z @ cx + mus / sig2s) ** 2 - mus ** 2 / sig2s) + 0.5 * np.log(betaxt / sig2s)
            # nuxt = nustar + 0.5 * (betaxt * (z @ cx + mus / sig2s) ** 2 - mus ** 2 / sig2s)  # no Gaussian space factor
            if prior:
                nuxt = nuxt + poisson.logpmf(pan + 1, mu=p1) - poisson.logpmf(pan, mu=p1)
            nuxt[np.isnan(nuxt)] = -np.inf
            # nuxt[t] = -np.inf
    nu_bk = nu.copy()
    nu = nu.T.flatten()

    indx = np.argsort(nu)[::-1]
    d_max = math.floor(indx[0] // P) + 1
    # num = min(math.ceil(np.sum(nu > nu.max() + np.log(psy_thresh))), D * P)
    num = min(30, D * P)
    nu_star = nu[indx[:num]]
    nu_star_bk = nu_bk.T.flatten()[indx[:num]]
    psy_star = np.exp(nu_star - nu.max()) / np.sum(np.exp(nu_star - nu.max()))
    T_star = [np.sort(T[:(indx[k] % P) + 1, indx[k] // P]) for k in range(num)]
    xmmse_star = np.empty((num, N))
    for k in range(num):
        xmmse_star[k] = xmmse[indx[k] % P, indx[k] // P]
    
    c_star = np.zeros_like(xmmse_star).astype(int)
    for k in range(num):
        t, c = np.unique(T_star[k][xmmse_star[k][T_star[k]] != 0], return_counts=True)
        c_star[k, t] = c

    if plot:
        cnorm = colors.Normalize(vmin=0, vmax=psy_star[0])
        cmap = cm.ScalarMappable(norm=cnorm, cmap=cm.Blues)
        cmap.set_array([])

        fig = plt.figure(figsize=(16, 16))
        fig.tight_layout()
        gs = gridspec.GridSpec(3, 2, figure=fig, left=0.1, right=0.9, top=0.95, bottom=0.1, wspace=0.2, hspace=0.2)
        ax = fig.add_subplot(gs[0, :])
        # cp = ax.imshow(nu_bk - nu_bk.min() + 1, aspect='auto', norm=colors.LogNorm())
        cp = ax.imshow(nu_bk, aspect='auto')
        fig.colorbar(cp, ax=ax)
        ax.set_xticks(np.arange(D))
        ax.set_xticklabels(np.arange(1, D + 1).astype(str))
        ax.set_yticks(np.arange(P))
        ax.set_yticklabels(np.arange(1, P + 1).astype(str))
        ax.set_xlabel('D')
        ax.set_ylabel('P')
        ax.scatter([ind // P for ind in indx[:num]], [ind % P for ind in indx[:num]], c=psy_star)
        ax.scatter(indx[0] // P, indx[0] % P, s=36.0, marker='o', facecolors='none', edgecolors='r')
        ax.hlines(len(truth) - 1, -0.5, D - 0.5, color='g')

        ax = fig.add_subplot(gs[1, 0])
        for k in range(1, num + 1):
            ax.plot(np.arange(1, P + 1), tlist[T[:, indx[num - k] // P]], c=cmap.to_rgba(psy_star[num - k]))
        fig.colorbar(cmap, ticks=np.arange(0, 1, 0.1))
        ax.scatter([ind % P + 1 for ind in indx[:num]], [tlist[T[indx[k] % P, indx[k] // P]] for k in range(num)], s=psy_star / psy_star[0] * 1000, marker='o', facecolors='none', edgecolors='r')
        ax.set_xticks(np.arange(1, P + 1))
        ax.set_xticklabels(np.arange(1, P + 1).astype(str))
        ax.set_xlabel('P')
        ax.set_ylabel('t/ns')

        ax = fig.add_subplot(gs[1, 1])
        ax.plot(np.arange(left, right), y, c='k')
        ax2 = ax.twinx()
        ax2.vlines(tlist, 0, xmmse[indx[0] % P, indx[0] // P] / mus, color='r')
        ax2.scatter(tlist, np.zeros_like(tlist), color='r')
        for t in T[:(indx[0] % P) + 1, indx[0] // P]:
            xx = np.zeros_like(xmmse[0, 0])
            xx[t] = xmmse[indx[0] % P, indx[0] // P][t]
            ax.plot(np.arange(left, right), np.dot(A, xx), 'r')
        for k in range(1, num + 1):
            ax.plot(np.arange(left, right), np.dot(A, xmmse[indx[num - k] % P, indx[num - k] // P]), c=cmap.to_rgba(psy_star[num - k]))
        ax2.plot(tlist, p1 / p1.max(), 'k--', alpha=0.5)
        ax.set_xlabel('t/ns')
        ax.set_ylabel('Voltage/V')
        ax2.set_ylabel('Charge/nsmV')
        xmin, xmax = ax.get_xlim()
        align.yaxes(ax, 0, ax2, 0)

        ax = fig.add_subplot(gs[2, 0])
        for k in range(1, num + 1):
            ax.vlines(tlist, 0, xmmse[indx[num - k] % P, indx[num - k] // P] / mus, color=cmap.to_rgba(psy_star[num - k]))
        fig.colorbar(cmap, ticks=np.arange(0, 1, 0.1))
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel('t/ns')
        ax.set_ylabel('Charge/nsmV')

        ax = fig.add_subplot(gs[2, 1])
        ax.plot(np.arange(left, right), y, c='b')
        ax2 = ax.twinx()
        ax2.vlines(truth['HitPosInWindow'], 0, truth['Charge'] / gmu, color='k')
        ax2.scatter(tlist, np.zeros_like(tlist), color='r')
        for t, c in zip(truth['HitPosInWindow'], truth['Charge']):
            ax.plot(t + np.arange(80), spe(np.arange(80), para[0], para[1], para[2]) * c / gmu, c='g')
        ax2.plot(tlist, p1 / p1.max(), 'k--', alpha=0.5)
        ax.set_xlabel('t/ns')
        ax.set_ylabel('Voltage/V')
        ax2.set_ylabel('Charge/nsmV')
        align.yaxes(ax, 0, ax2, 0)
        fig.savefig('t/' + str(i) + '.png')
        plt.close()

    xmmse = np.average(xmmse_star, weights=psy_star, axis=0)

    return xmmse, xmmse_star, psy_star, nu_star, nu_star_bk, T_star, d_max, num

def nu_direct(y, A, nx, mus, sig2s, sig2w, la, prior=True):
    M, N = A.shape
    Phi = np.matmul(np.matmul(A, np.diagflat(sig2s * nx)), A.T) + np.eye(M) * sig2w
    z = y - np.dot(A, (mus * nx))
    invPhi = np.linalg.inv(Phi)
    detPhi = np.linalg.det(Phi)
    nu = -0.5 * np.matmul(np.matmul(z, invPhi), z) - 0.5 * M * np.log(2 * np.pi) - 0.5 * np.log(detPhi)
    # nu = -0.5 * np.matmul(np.matmul(z, invPhi), z) - 0.5 * M * np.log(2 * np.pi)  # no Gaussian space factor
    if prior:
        nu = nu + poisson.logpmf(nx, mu=la).sum()
    return nu

def elbo(nu_star_prior):
    q = np.exp(nu_star_prior - nu_star_prior.max()) / np.sum(np.exp(nu_star_prior - nu_star_prior.max()))
    e = np.sum(q * nu_star_prior) - np.sum(q * np.log(q))
    e_star = special.logsumexp(nu_star_prior)
    assert abs(e_star - e) < 1e-6
    return e

def rss_alpha(alpha, outputs, inputs, mnecpu):
    r = np.power(alpha * np.matmul(mnecpu, outputs) - inputs, 2).sum()
    return r

def shannon_interpolation(w, n):
    t = np.arange(0, len(w), 1 / n)
    l = np.arange(len(w))
    y = np.sum(np.sinc(t[:, None] - l) * w, axis=1)
    return y

def read_model(spe_path, n=1):
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
            # mar_l = 0
            # mar_r = 0
            ax.plot(spe[i])
            spe_pre_i = {'spe':interp1d(np.arange(len(spe[i])), spe[i])(np.arange(0, len(spe[i]) - 1, 1 / n)), 'epulse':epulse, 'peak_c':peak_c * n, 'mar_l':mar_l * n, 'mar_r':mar_r * n, 'std':std[i], 'parameters':p[i]}
            spe_pre.update({cid[i]:spe_pre_i})
        ax.grid()
        ax.set_xlabel(r'$\mathrm{Time}/\si{ns}$')
        ax.set_ylabel(r'$\mathrm{Voltage}/\si{mV}$')
        # fig.savefig('Note/figures/pmtspe.pdf')
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

def spe(t, tau, sigma, A, gmu=gmu, window=window):
    return A * np.exp(-1 / 2 * (np.log(t / tau) / sigma) ** 2)
    # return np.where(t == 0, gmu, 0)
    # return np.ones_like(t) / window * gmu

def charge(n, gmu, gsigma, thres=0):
    chargesam = norm.ppf(1 - uniform.rvs(scale=1-norm.cdf(thres, loc=gmu, scale=gsigma), size=n), loc=gmu, scale=gsigma)
    return chargesam

def probcharhitt(t0, hitt, probcharge, Tau, Sigma, npe):
    prob = np.where(npe >= 0, probcharge * np.power(convolve_exp_norm(hitt - t0, Tau, Sigma), npe), 0)
    prob = np.sum(prob, axis=1) / np.sum(probcharge, axis=1)
    return prob

def npeprobcharge(charge, npe, gmu, gsigma, s0):
    scale = np.where(npe != 0, gsigma * np.sqrt(npe), gsigma * np.sqrt(s0))
    prob = np.where(npe >= 0, norm.pdf(charge, loc=gmu * npe, scale=scale) / (1 - norm.cdf(0, loc=gmu * npe, scale=scale)), 0)
    return prob

def likelihoodt0(hitt, char, gmu, Tau, Sigma, mode='charge', is_delta=False):
    b = [0., 600.]
    tlist = np.arange(b[0], b[1] + 1e-6, 0.2)
    if mode == 'charge':
        logL = lambda t0 : -1 * np.sum(np.log(np.clip(convolve_exp_norm(hitt - t0, Tau, Sigma), np.finfo(np.float64).tiny, np.inf)) * char / gmu)
    elif mode == 'all':
        logL = lambda t0 : -1 * np.sum(np.log(np.clip(convolve_exp_norm(hitt - t0, Tau, Sigma), np.finfo(np.float64).tiny, np.inf)))
    logLv_tlist = np.vectorize(logL)(tlist)
    t0 = opti.fmin_l_bfgs_b(logL, x0=[tlist[np.argmin(logLv_tlist)]], approx_grad=True, bounds=[b], maxfun=500000)[0]
    t0delta = None
    if is_delta:
        logLvdelta = np.vectorize(lambda t : np.abs(logL(t) - logL(t0) - 0.5))
        t0delta = abs(opti.fmin_l_bfgs_b(logLvdelta, x0=[tlist[np.argmin(np.abs(logLv_tlist - logL(t0) - 0.5))]], approx_grad=True, bounds=[b], maxfun=500000)[0] - t0)
    return t0, t0delta

def initial_params(wave, spe_pre, Tau, Sigma, gmu, Thres, p, nsp, nstd, is_t0=False, is_delta=False, n=1, nshannon=1):
    hitt, char = lucyddm(wave[::nshannon], spe_pre['spe'][::nshannon])
    hitt, char = clip(hitt, char, Thres)
    char = char / char.sum() * np.clip(np.abs(wave[::nshannon].sum()), 1e-6, np.inf)
    tlist = np.unique(np.clip(np.hstack(hitt[:, None] + np.arange(-nsp, nsp+1)), 0, len(wave[::nshannon]) - 1))

    index_prom = np.hstack([np.argwhere(savgol_filter(wave, 11 * (nshannon if nshannon % 2 == 1 else nshannon + 1), 4) > nstd * spe_pre['std']).flatten(), hitt * nshannon])
    left_wave = round(np.clip(index_prom.min() - 3 * spe_pre['mar_l'], 0, len(wave) - 1))
    right_wave = round(np.clip(index_prom.max() + 3 * spe_pre['mar_r'], 0, len(wave) - 1))
    wave = wave[left_wave:right_wave]

    npe_init = np.zeros(len(tlist))
    npe_init[np.isin(tlist, hitt)] = char / gmu
    npe_init = np.repeat(npe_init, n) / n
    tlist = np.unique(np.sort(np.hstack(tlist[:, None] + np.linspace(0, 1, n, endpoint=False) - (n // 2) / n)))
    if len(tlist) != 1:
        assert abs(np.diff(tlist).min() - 1 / n) < 1e-3, 'tlist anomalous'
    t_auto = (np.arange(left_wave, right_wave) / nshannon)[:, None] - tlist
    A = spe((t_auto + np.abs(t_auto)) / 2, p[0], p[1], p[2])

    t0_init = None
    t0_init_delta = None
    if is_t0:
        t0_init, t0_init_delta = likelihoodt0(hitt=hitt, char=char, gmu=gmu, Tau=Tau, Sigma=Sigma, mode='charge', is_delta=is_delta)
    return A, wave, tlist, t0_init, t0_init_delta, npe_init, left_wave, right_wave

def stdrmoutlier(array, r):
    arrayrmoutlier = array[np.abs(array - np.mean(array)) < r * np.std(array, ddof=-1)]
    std = np.std(arrayrmoutlier, ddof=-1)
    return std, len(arrayrmoutlier)

def demo(pet_sub, cha_sub, tth, spe_pre, window, wave, cid, p, full=False, fold='Note/figures', ext='.pgf'):
    penum = len(tth)
    print('PEnum is {}'.format(penum))
    pan = np.arange(window)
    pet_ans_0 = tth['HitPosInWindow']
    cha_ans = tth['Charge'] / spe_pre['spe'].sum()
    pet_ans = np.unique(pet_ans_0)
    cha_ans = np.array([np.sum(cha_ans[pet_ans_0 == i]) for i in pet_ans])
    ylabel = r'$\mathrm{Charge}$'
    distd = '(W/ns,C/mV*ns)'
    distl = 'cdiff'
    edist = (cha_sub.sum() - cha_ans.sum()) * spe_pre['spe'].sum()
    print('truth HitPosInWindow = {}, Weight = {}'.format(pet_ans, cha_ans))
    wav_ans = np.sum([np.where(pan > pet_ans[j], spe(pan - pet_ans[j], tau=p[0], sigma=p[1], A=p[2]) * cha_ans[j], 0) for j in range(len(pet_ans))], axis=0)
    print('truth RSS = {}'.format(np.power(wave - wav_ans, 2).sum()))
    print('HitPosInWindow = {}, Weight = {}'.format(pet_sub, cha_sub))
    wdist = scipy.stats.wasserstein_distance(pet_ans, pet_sub, u_weights=cha_ans, v_weights=cha_sub)
    print('wdist = {}, '.format(wdist) + distl + ' = {}'.format(edist))
    wav_sub = np.sum([np.where(pan > pet_sub[j], spe(pan - pet_sub[j], tau=p[0], sigma=p[1], A=p[2]) * cha_sub[j], 0) for j in range(len(pet_sub))], axis=0)
    print('RSS = {}'.format(np.power(wav_ans - wav_sub, 2).sum()))

    fig = plt.figure(figsize=(10, 10))
    fig.tight_layout()
    ax0 = fig.add_axes((.1, .2, .85, .3))
    ax0.plot(pan, wave, c='b', label='origin wave')
    ax0.plot(pan, wav_ans, c='k', label='truth wave')
    ax0.plot(pan, wav_sub, c='g', label='recon wave')
    ax0.set_ylabel('$\mathrm{Voltage}/\si{mV}$')
    ax0.hlines(5 * spe_pre['std'], 0, window, color='c', label='threshold')
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
    ax1.set_yticks(np.arange(0, max(max(cha_ans), max(cha_sub)), 0.2))
    ax1.legend(loc=1)
    ax1.grid()
    ax2 = fig.add_axes((.1, .7, .85, .2))
    ax2.vlines(pet_sub, 0, cha_sub, color='g', label='recon Charge')
    ax2.set_ylabel(ylabel)
    ax2.set_xticklabels([])
    ax2.set_xlim(ax0.get_xlim())
    ax2.set_ylim(0, max(max(cha_ans), max(cha_sub))*1.1)
    ax2.set_yticks(np.arange(0, max(max(cha_ans), max(cha_sub)), 0.2))
    ax2.legend(loc=1)
    ax2.grid()
    ax3 = fig.add_axes((.1, .1, .85, .1))
    ax3.scatter(pan, wav_sub - wave, c='k', label='residual wave', marker='.')
    ax3.set_xlabel('$\mathrm{t}/\si{ns}$')
    ax3.set_ylabel('$\mathrm{Voltage}/\si{mV}$')
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
