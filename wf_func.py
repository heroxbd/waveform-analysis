from operator import le
import os
import math
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

import numpy as np
np.set_printoptions(suppress=True)
import scipy
import scipy.stats
from scipy.stats import poisson, uniform, norm, gamma
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
from numba import njit
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
gsigma = 160. * 0.4
std = 1.
p = [8., 0.5, 24.]
Thres = {'mcmc':std / gsigma, 'xiaopeip':0, 'lucyddm':0.2, 'fsmp':0, 'fftrans':0.1, 'findpeak':0.1, 'threshold':0, 'firstthres':0, 'omp':0}
d_history = [('TriggerNo', np.uint32), ('ChannelID', np.uint32), ('step', np.uint32), ('loc', np.float32)]
proposal = np.array((1, 1, 2)) / 4

def xiaopeip_old(wave, spe_pre, eta=0):
    l = len(wave)
    flag = 1
    lowp = np.argwhere(wave > 5 * spe_pre['std']).flatten()
    if len(lowp) != 0:
        fitp = np.arange(lowp.min() - round(spe_pre['mar_l']), lowp.max() + round(spe_pre['mar_r']))
        fitp = np.unique(np.clip(fitp, 0, len(wave)-1))
        pet = lowp - spe_pre['peak_c']
        pet = np.unique(np.clip(pet, 0, len(wave)-1))
        if len(pet) != 0:
            # cha, ped = xiaopeip_core(wave, spe_pre['spe'], fitp, pet, eta=eta)
            cha = xiaopeip_core(wave[fitp], spe_pre['spe'], fitp, pet, eta=eta)
        else:
            flag = 0
    else:
        flag = 0
    if flag == 0:
        pet = np.array([np.argmax(wave[spe_pre['peak_c']:])])
        cha = np.array([1])
    # return pet, cha, ped
    return pet, cha

def xiaopeip(wave, spe_pre, Tau, Sigma, Thres, p, eta=0):
    '''
    eta is the hyperparameter level of LASSO passed to xiaopeip_core.
    '''
    _, wave_r, tlist, _, _, _, left_wave, right_wave = initial_params(wave, spe_pre, Tau, Sigma, gmu, Thres, p, is_t0=False, is_delta=False, n=1)
    fitp = np.arange(left_wave, right_wave)
    # cha, ped = xiaopeip_core(wave_r, spe_pre['spe'], fitp, tlist.astype(int), eta=eta)
    cha = xiaopeip_core(wave_r, spe_pre['spe'], fitp, tlist.astype(int), eta=eta)
    return tlist, cha

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

def xiaopeip_core(wave_r, spe, fitp, possible, eta=0):
    l = window
    spe = np.concatenate([spe, np.zeros(l - len(spe))])
    ans0 = np.ones(len(possible)).astype(np.float64)
    b = np.zeros((len(possible), 2)).astype(np.float64)
    b[:, 1] = np.inf
    mne = spe[np.mod(fitp.reshape(len(fitp), 1) - possible.reshape(1, len(possible)), l)]
    try:
        ans = opti.fmin_l_bfgs_b(norm_fit, ans0, args=(mne, wave_r, eta), approx_grad=True, bounds=b, maxfun=500000)
    except ValueError:
        ans = [np.ones(len(possible)) * 0.2]
    # ans = opti.fmin_slsqp(norm_fit, ans0, args=(mne, wave_r), bounds=b, iprint=-1, iter=500000)
    # ans = opti.fmin_tnc(norm_fit, ans0, args=(mne, wave_r), approx_grad=True, bounds=b, messages=0, maxfun=500000)
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

def firstthres(wave, spe_pre):
    pet = np.argwhere(wave[spe_pre['peak_c']:] > spe_pre['std'] * 5).flatten()
    if len(pet) == 0:
        pet = np.array([np.argmax(wave[spe_pre['peak_c']:])])
    else:
        pet = pet[:1]
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

def combine(A, cx, t):
    '''
    combine neighbouring dictionaries to represent sub-bin locations
    '''
    frac, ti = np.modf(t - 0.5)
    ti = int(ti)
    alpha = np.array((1 - frac, frac))
    return alpha @ A[:, ti:(ti+2)].T, alpha @ cx[:, ti:(ti+2)].T

def move(A_vec, c_vec, z, step, mus, sig2s, A):
    '''
    A_vec: 行向量
    c_vec: 行向量

    step
    ====
    1: 在 t 加一个 PE
    -1: 在 t 减一个 PE
    '''
    fsig2s = step * sig2s
    # Eq. (30) sig2s = 1 sigma^2 - 0 sigma^2
    beta_under = (1 + fsig2s * np.dot(A_vec, c_vec))
    beta = fsig2s / beta_under

    # Eq. (31) # sign of mus[t] and sig2s[t] cancels
    Δν = 0.5 * (beta * (z @ c_vec + mus / sig2s) ** 2 - mus ** 2 / fsig2s)
    # sign of space factor in Eq. (31) is reversed.  Because Eq. (82) is in the denominator.
    Δν -= 0.5 * np.log(beta_under) # space
    # accept, prepare for the next
    # Eq. (33) istar is now n_pre.  It crosses n_pre and n, thus is in vector form.
    Δcx = -np.einsum('n,m,mp->np', beta * c_vec, c_vec, A, optimize=True)

    # Eq. (34)
    Δz = -step * A_vec * mus
    return Δν, Δcx, Δz

def flow(cx, p1, z, N, sig2s, sig2w, mus, A, p_cha, mu_t, TRIALS=2000):
    '''
    flow
    ====
    连续时间游走
    cx: Cov^-1 * A, 详见 FBMP
    s: list of PE locations
    mu_t: LucyDDM 的估算 PE 数
    z: residue waveform
    '''
    # istar [0, 1) 之间的随机数，用于点中 PE
    istar = np.random.rand(TRIALS)
    # 同时可用于创生位置的选取
    c_cha = np.cumsum(p_cha) # cha: charge; p_cha: pdf of LucyDDM charge (由 charge 引导的 PE 强度流先验)
    home_s = np.interp(istar, xp=np.insert(c_cha, 0, 0), fp=np.arange(N+1)) # 根据 p_cha 采样得到的 PE 序列。供以后的产生过程使用。这两行是使用了 InverseCDF 算法进行的MC采样。

    NPE0 = int(mu_t + 0.5) # mu_t: μ_total，LucyDDM 给出的 μ 猜测；NPE0 是 PE 序列初值 s_0 的 PE 数。
    # t 的位置，取值为 [0, N)
    s = list(np.interp((np.arange(NPE0) + 0.5) / NPE0, xp=np.insert(c_cha, 0, 0), fp=np.arange(N+1))) # MCMC 链的 PE configuration 初值 s0
    ν = 0
    for t in s: # 从空序列开始逐渐加 PE 以计算 s0 的 ν, cx, z
        Δν, Δcx, Δz = move(*combine(A, cx, t), z, 1, mus, sig2s, A)
        ν += Δν
        cx += Δcx
        z += Δz

    # s 的记录方式：使用定长 compound array es_history 存储(存在 'loc' 里)，但由于 s 实际上变长，每一个有相同  'step' 的 'loc' 属于一个 s，si 作为临时变量用于分割成不定长片段，每一段是一个 s。
    si = 0
    es_history = np.zeros(TRIALS * (NPE0 + 5) * N, dtype=d_history)

    wander_s = np.random.normal(size=TRIALS)

    # step: +1 创生一个 PE， -1 消灭一个 PE， +2 向左或向右移动
    flip = np.random.choice((-1, 1, 2), TRIALS, p=proposal)
    Δν_history = np.zeros(TRIALS) # list of Δν's
    log_mu = np.log(mu_t) # 猜测的 Poisson 流强度
    T_list = []
    c_star_list = []

    for i, (t, step, home, wander, accept) in enumerate(zip(istar, flip, home_s, wander_s, np.log(np.random.rand(TRIALS)))):
        # 不设左右边界
        NPE = len(s)
        if NPE == 0:
            step = 1 # 只能创生
            accept += np.log(1 / proposal[1]) # 惩罚
        elif NPE == 1 and step == -1:
            # 1 -> 0: 行动后从 0 脱出的几率大，需要鼓励
            accept -= np.log(1 / proposal[0])

        if step == 1: # 创生
            if home >= 0.5 and home <= N - 0.5: 
                Δν, Δcx, Δz = move(*combine(A, cx, home), z, 1, mus, sig2s, A)
                Δν += log_mu - np.log(NPE + 1)
                if Δν >= accept:
                    s.append(home)
            else: # p(w|s) 无定义
                Δν = -np.inf
        else:
            op = int(t * NPE) # 操作的 PE 编号
            loc = s[op] # 待操作 PE 的位置
            Δν, Δcx, Δz = move(*combine(A, cx, loc), z, -1, mus, sig2s, A)
            if step == -1: # 消灭
                Δν -= log_mu - np.log(NPE)
                if Δν >= accept:
                    del s[op]
            elif step == 2: # 移动
                nloc = loc + wander # 待操作 PE 的新位置
                if nloc >= 0.5 and nloc <= N - 0.5: # p(w|s) 无定义
                    Δν1, Δcx1, Δz1 = move(*combine(A, cx + Δcx, nloc), z + Δz, 1, mus, sig2s, A)
                    Δν += Δν1
                    Δν += np.log(p_cha[int(nloc)]) - np.log(p_cha[int(loc)])
                    if Δν >= accept:
                        s[op] = nloc
                        Δcx += Δcx1
                        Δz += Δz1
                else: # p(w|s) 无定义
                    Δν = -np.inf

        if Δν >= accept:
            cx += Δcx
            z += Δz
            si1 = si + len(s)
            es_history[si:si1]['step'] = i
            es_history[si:si1]['loc'] = s
            si = si1
        else: # reject proposal
            Δν = 0
            step = 0
        T_list.append(np.sort(np.digitize(s, bins=np.arange(N)) - 1))
        t, c = np.unique(T_list[-1], return_counts=True)
        c_star = np.zeros(N, dtype=int)
        c_star[t] = c
        c_star_list.append(c_star)

        Δν_history[i] = Δν
        flip[i] = step
    return flip, [Δν_history, ν], es_history[:si1], c_star_list, T_list

def metropolis_fsmp(y, A, sig2w, sig2s, mus, p1, p_cha, mu_t, TRIALS=2000):
    '''
    p1: prior probability for each bin.
    sig2w: variance of white noise.
    sig2s: variance of signal x_i.
    mus: mean of signal x_i.
    '''
    # Only for multi-gaussian with arithmetic sequence of mu and sigma
    M, N = A.shape

    # nu_root: nu for all s_n=0.
    nu_root = -0.5 * np.linalg.norm(y) ** 2 / sig2w - 0.5 * M * np.log(2 * np.pi)
    nu_root -= 0.5 * M * np.log(sig2w)
    nu_root += poisson.logpmf(0, p1).sum()
    # Eq. (29)
    cx_root = A / sig2w
    # mu = 0 => (y - A * mu -> z)
    z = y.copy()

    # Metropolis flow
    flip, Δν_history, es_history, c_star_list, T_list = flow(cx_root, p1, z, N, sig2s, sig2w, mus, A, p_cha, mu_t, TRIALS=TRIALS)
    num = len(T_list)

    c_star = np.vstack(c_star_list)
    nu_star = np.cumsum(Δν_history[0]) + nu_root + Δν_history[1]

    burn = num // 5
    nu_star = nu_star[burn:]
    T_list = T_list[burn:]
    c_star = c_star[burn:, :]
    flip[np.abs(flip) == 2] = 0 # 平移不改变 PE 数
    NPE_evo = np.cumsum(np.insert(flip, 0, int(mu_t + 0.5)))[burn:]
    es_history = es_history[es_history['step'] >= burn]

    return nu_star, T_list, c_star, es_history, NPE_evo

def nu_direct(y, A, nx, mus, sig2s, sig2w, la):
    M, N = A.shape
    Phi_s = Phi(y, A, nx, mus, sig2s, sig2w)
    z = y - np.dot(A, (mus * nx))
    invPhi = np.linalg.inv(Phi_s)
    nu = -0.5 * np.matmul(np.matmul(z, invPhi), z) - 0.5 * M * np.log(2 * np.pi)
    nu -= 0.5 * np.log(np.linalg.det(Phi_s))
    nu = nu + poisson.logpmf(nx, mu=la).sum()
    return nu

def Phi(y, A, nx, mus, sig2s, sig2w):
    M, N = A.shape
    return np.matmul(np.matmul(A, np.diagflat(sig2s * nx)), A.T) + np.eye(M) * sig2w

def elbo(nu_star_prior):
    q = np.exp(nu_star_prior - nu_star_prior.max()) / np.sum(np.exp(nu_star_prior - nu_star_prior.max()))
    e = np.sum(q * nu_star_prior) - np.sum(q * np.log(q))
    # e_star = special.logsumexp(nu_star_prior)
    # assert abs(e_star - e) < 1e-4
    return e

@njit(nogil=True, cache=True)
def unique_with_indices(values):
    unq = np.unique(values)
    idx = np.zeros_like(unq, dtype=np.int_)
    idx[0] = 0
    i = 0
    for j in range(1, len(values)):
        if values[j] != unq[i]:
            i += 1
            idx[i] = j
    return unq, idx

@njit(nogil=True, cache=True)
def group_by_sorted_count_sum(idx, a):
    unique_idx, idx_of_idx = unique_with_indices(idx)
    counts = np.zeros_like(unique_idx, dtype=np.int_)
    sums = np.zeros_like(unique_idx, dtype=np.float64)
    for i in range(0, len(idx_of_idx)):
        start = idx_of_idx[i]
        if i < len(idx_of_idx) - 1:
            end = idx_of_idx[i + 1]
        else:
            end = len(idx)
        counts[i] = end - start
        sums[i] = np.sum(a[start:end])
    return unique_idx, counts, sums

@njit(nogil=True, cache=True)
def jit_logsumexp(values, b):
    a_max = np.max(values)
    s = np.sum(b * np.exp(values - a_max))
    return np.log(s) + a_max

@njit(nogil=True, cache=True)
def group_by_logsumexp(idx, a, b):
    unique_idx, idx_of_idx = unique_with_indices(idx)
    res = np.zeros_like(unique_idx, dtype=np.float64)
    for i in range(0, len(idx_of_idx)):
        start = idx_of_idx[i]
        if i < len(idx_of_idx) - 1:
            end = idx_of_idx[i + 1]
        else:
            end = len(idx)
        res[i] = jit_logsumexp(a[start:end], b[start:end])
    return unique_idx, res

def jit_agg_NPE(step, f, size):
    step, NPE, f_vec = group_by_sorted_count_sum(step, f)

    f_vec_merged = np.zeros(
        len(step),
        dtype=np.dtype([("NPE", np.int_), ("f_vec", np.float64), ("repeat", np.int_)]),
    )
    f_vec_merged["NPE"] = NPE
    f_vec_merged["f_vec"] = f_vec
    f_vec_merged["repeat"] = np.diff(np.append(step, int(size)))

    f_vec_merged = np.sort(f_vec_merged, order="NPE")

    indices, NPE_vec = group_by_logsumexp(
        f_vec_merged["NPE"], f_vec_merged["f_vec"], f_vec_merged["repeat"]
    )

    return indices, NPE_vec

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
    if tau == 0.0:
        y = norm.pdf(x, loc=0, scale=sigma)
    elif sigma == 0.0:
        y = np.where(x >= 0.0, 1.0 / tau * np.exp(-x / tau), 0.0)
    else:
        alpha = 1 / tau
        co = alpha / 2.0 * np.exp(alpha * alpha * sigma * sigma / 2.0)
        x_erf = (alpha * sigma * sigma - x) / (np.sqrt(2.) * sigma)
        y = co * (1.0 - special.erf(x_erf)) * np.exp(-alpha * x)
    return y

def log_convolve_exp_norm(x, tau, sigma):
    if tau == 0.0:
        y = norm.logpdf(x, loc=0, scale=sigma)
    elif sigma == 0.0:
        y = np.where(x >= 0.0, -np.log(tau) - x / tau, -np.inf)
    else:
        alpha = 1.0 / tau
        co = -np.log(2.0 * tau) + alpha * alpha * sigma * sigma / 2.0
        x_erf = (alpha * sigma * sigma - x) / (np.sqrt(2.0) * sigma)
        y = co + np.log(1.0 - special.erf(x_erf)) - alpha * x
    return np.clip(y, np.log(np.finfo(np.float64).tiny), np.inf)

def spe(t, tau, sigma, A, gmu=gmu, window=window):
    return A * np.exp(-1 / 2 * (np.log(t / tau) / sigma) ** 2)
    # return np.where(t == 0, gmu, 0)
    # return np.ones_like(t) / window * gmu

def charge(n, gmu, gsigma, thres=0):
    chargesam = gamma.rvs(a=(gmu / gsigma) ** 2, loc=0, scale=gsigma**2/gmu, size=n)
    # alpha = (gmu / gsigma) ** 2
    # beta = gmu / gsigma ** 2
    # chargesam = gamma.rvs(a=alpha, loc=0, scale=1/beta, size=n)
    # chargesam = norm.rvs(loc=gmu, scale=gsigma, size=n)
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

def initial_params(wave, spe_pre, Tau, Sigma, gmu, Thres, p, nsp=4, nstd=3, is_t0=False, is_delta=False, n=1):
    hitt_r, char_r = lucyddm(wave, spe_pre['spe'])
    hitt_r, char_r = clip(hitt_r, char_r, Thres)
    hitt = np.arange(hitt_r.min(), hitt_r.max() + 1)
    hitt[np.isin(hitt, hitt_r)] = hitt_r
    char = np.zeros(len(hitt))
    char[np.isin(hitt, hitt_r)] = char_r
    char = char / char.sum() * np.clip(np.abs(wave.sum()), 1e-6, np.inf)
    tlist = np.unique(np.clip(np.hstack(hitt[:, None] + np.arange(-nsp, nsp+1)), 0, len(wave) - 1))

    index_prom = np.hstack([np.argwhere(savgol_filter(wave, 11, 4) > nstd * spe_pre['std']).flatten(), hitt])
    left_wave = round(np.clip(index_prom.min() - 3 * spe_pre['mar_l'], 0, len(wave) - 1))
    right_wave = round(np.clip(index_prom.max() + 3 * spe_pre['mar_r'], 0, len(wave) - 1))
    wave = wave[left_wave:right_wave]

    npe_init = np.zeros(len(tlist))
    npe_init[np.isin(tlist, hitt)] = char / gmu
    npe_init = np.repeat(npe_init, n) / n
    tlist = np.unique(np.sort(np.hstack(tlist[:, None] + np.linspace(0, 1, n, endpoint=False) - (n // 2) / n)))
    if len(tlist) != 1:
        assert abs(np.diff(tlist).min() - 1 / n) < 1e-3, 'tlist anomalous'
    t_auto = np.arange(left_wave, right_wave)[:, None] - tlist
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
    ax0.plot(pan, wave, c='b', label='noisy waveform')
    ax0.plot(pan, wav_ans, c='k', label='true waveform')
    ax0.plot(pan, wav_sub, c='g', label='reconstructed waveform')
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
    ax1.vlines(pet_ans, 0, cha_ans, color='k', label='true charge')
    ax1.set_ylabel(ylabel)
    ax1.set_xticklabels([])
    ax1.set_xlim(ax0.get_xlim())
    ax1.set_ylim(0, max(max(cha_ans), max(cha_sub))*1.1)
    ax1.set_yticks(np.arange(0, max(max(cha_ans), max(cha_sub)), 0.2))
    ax1.legend(loc=1)
    ax1.grid()
    ax2 = fig.add_axes((.1, .7, .85, .2))
    ax2.vlines(pet_sub, 0, cha_sub, color='g', label='reconstructed charge')
    ax2.set_ylabel(ylabel)
    ax2.set_xticklabels([])
    ax2.set_xlim(ax0.get_xlim())
    ax2.set_ylim(0, max(max(cha_ans), max(cha_sub))*1.1)
    ax2.set_yticks(np.arange(0, max(max(cha_ans), max(cha_sub)), 0.2))
    ax2.legend(loc=1)
    ax2.grid()
    ax3 = fig.add_axes((.1, .1, .85, .1))
    ax3.scatter(pan, wav_sub - wave, c='k', label='residuals', marker='.')
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
