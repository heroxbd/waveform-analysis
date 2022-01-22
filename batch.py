#!/usr/bin/env python3
import time
import argparse

import h5py
import numpy as np

# np.seterr(all='raise')
from scipy.special import erf
from scipy.stats import norm

import itertools as it

psr = argparse.ArgumentParser()
psr.add_argument("-o", dest="opt", type=str, help="output file")
psr.add_argument("ipt", type=str, help="input file")
psr.add_argument("--size", type=int, default=100, help="batch size")
args = psr.parse_args()

fipt = args.ipt
fopt = args.opt

dt = [
    ("TriggerNo", np.uint32),
    ("ChannelID", np.uint32),
    ("flip", np.int8),
    ("delta_nu", np.float64),
    ("t0", np.float64),
]
mu0_dt = [("TriggerNo", np.uint32), ("ChannelID", np.uint32), ("mu_t", np.float64)]

d_history = [
    ("TriggerNo", np.uint32),
    ("ChannelID", np.uint32),
    ("step", np.uint32),
    ("loc", np.float32),
]
TRIALS = 5000

def combine(A, cx, t):
    """
    combine neighbouring dictionaries to represent sub-bin locations
    """
    frac, ti = np.modf(t - 0.5)
    ti = np.array(ti, np.int32)
    w_all = np.arange(A.shape[0]) # index of all the waveforms
    A_vec = (1 - frac)[:, None] * A[w_all, :, ti] + frac[:, None] * A[w_all, :, ti+1]
    c_vec = (1 - frac)[:, None] * cx[w_all, :, ti] + frac[:, None] * cx[w_all, :, ti+1]
    return A_vec, c_vec

def lc(x, tau=20, sigma=5):
    """
    light curve
    """
    alpha = 1 / tau
    co = -np.log(2.0 * tau) + alpha * alpha * sigma * sigma / 2.0
    x_erf = (alpha * sigma * sigma - x) / (np.sqrt(2.0) * sigma)
    return co + np.log(1.0 - erf(x_erf)) - alpha * x


def move1(A_vec, c_vec, z, step, mus, sig2s):
    """
    step
    ====
    A_vec: 行向量
    c_vec: 行向量
    z: 残余波形
    step:
        1: 在 t 加一个 PE
        -1: 在 t 减一个 PE
    mus: spe波形的平均幅值
    sig2s: spe波形幅值方差
    A: spe, PE x waveform
    """
    fsig2s = step * sig2s
    # Eq. (30) sig2s = 1 sigma^2 - 0 sigma^2
    beta_under = 1 + fsig2s * np.einsum('ij,ij->i', A_vec, c_vec)
    beta = fsig2s / beta_under

    # Eq. (31) # sign of mus[t] and sig2s[t] cancels
    Δν = 0.5 * (beta * (np.einsum('ij,ij->i', z, c_vec) + mus / sig2s) ** 2 - mus ** 2 / fsig2s)
    # sign of space factor in Eq. (31) is reversed.  Because Eq. (82) is in the denominator.
    Δν -= 0.5 * np.log(beta_under)  # space
    return Δν, beta

def move2(A_vec, c_vec, step, mus, A, beta):
    # accept, prepare for the next
    # Eq. (33) istar is now n_pre.  It crosses n_pre and n, thus is in vector form.
    Δcx = -np.einsum("e,en,em,emp->enp", beta, c_vec, c_vec, A, optimize=True)

    # Eq. (34)
    Δz = -(step * mus)[:, None] * A_vec
    return Δcx, Δz

def move(A_vec, c_vec, z, step, mus, sig2s, A):
    Δν, beta = move1(A_vec, c_vec, z, step, mus, sig2s)
    return Δν, *move2(A_vec, c_vec, step, mus, A, beta)

def periodic(_h, _lt):
    '''
    enforce a periodic boundary at 0.5 and _lt - 0.5
    '''
    c_overflow = _h < 0.5
    _h[c_overflow] += _lt[c_overflow] - 1
    c_overflow = _h > _lt - 0.5
    _h[c_overflow] -= _lt[c_overflow] - 1
    return _h

with h5py.File(fipt, "r", libver="latest", swmr=True) as ipt:
    A = ipt["A"][:1000]
    index = ipt["index"][:1000]
    tq = ipt["tq"][:1000]
    z = ipt["z"][:1000]

@profile
def batch(A, index, tq, z):
    """
    batch
    ====
    连续时间游走
    cx: Cov^-1 * A, 详见 FBMP
    s: list of PE locations
    mu_t: LucyDDM 的估算 PE 数
    z: residue waveform
    """
    cx = (A / index["sig2w"][:, None, None])

    l_e = len(index)
    l_t = tq.shape[1]
    # istar [0, 1) 之间的随机数，用于点中 PE
    istar = np.random.rand(l_e, TRIALS)
    # 同时可用于创生位置的选取
    # q_s: pdf of LucyDDM charge (由 charge 引导的 PE 强度流先验)
    cq = np.cumsum(tq["q_s"], axis=1)
    cq0 = np.insert(cq, 0, 0, axis=1)

    # 根据 p_cha 采样得到的 PE 序列。供以后的产生过程使用。这两行是使用了 InverseCDF 算法进行的MC采样。
    fp = np.arange(l_t)
    home_s = np.array([ np.interp(_is, xp=_xp, fp=fp) for _is, _xp in zip(istar, cq0)])

    NPE0 = np.round(index["mu0"])  # mu_t: μ_total，LucyDDM 给出的 μ 猜测；NPE0 是 PE 序列初值 s_0 的 PE 数。
    # t 的位置，取值为 [0, N)
    # MCMC 链的 PE configuration 初值 s0
    s = [ list(
        np.interp(
            (np.arange(_npe) + 0.5) / _npe,
            xp = _xp,
            fp = fp,
        )
    ) for _npe, _xp in zip(NPE0, cq0) ]

    # 从空序列开始逐渐加 PE 以计算 s0 的 ν, cx, z
    for _s in it.zip_longest(*s):
        e = np.where(_s)[0] # waveform index
        A_vec, c_vec = combine(A[e], cx[e], np.array(_s, np.float32)[e])
        Δν, Δcx, Δz = move(A_vec, c_vec, z[e], 1, index["mus"][e], index["sig2s"][e], A[e])
        cx[e] += Δcx
        z[e] += Δz

    t0 = []
    p1 = []
    for _s, _lt, _ts in zip(s, index["l_t"], tq["t_s"]):
        if len(_s) == 0:
            t0.append(tlist[0])
            p1.append([])
        else:
            real_t = np.interp(_s, xp=np.arange(0.5, _lt), fp=_ts[:_lt])
            _t0 = real_t[0]
            t0.append(_t0)
            p1.append(lc(real_t - _t0))
    t0 = np.array(t0, np.float32)

    # step: +1 创生一个 PE， -1 消灭一个 PE， +2 向左或向右移动
    flip = np.random.choice((-1, 1, 2), (l_e, TRIALS), p=np.array((1, 1, 2)) / 4)
    Δν_history = np.zeros((l_e, TRIALS)) # list of Δν's
    t0_history = np.zeros((l_e, TRIALS))
    s0_history = np.zeros((l_e, TRIALS), dtype=np.int32) # 0-norm of s

    log_mu = np.log(index["mu0"])  # 猜测的 Poisson 流强度
    loc = np.zeros(l_e)

    NPE = np.array([len(_s) for _s in s])
    for i, (t, step, home, wander, wt, accept, acct) in enumerate(
        zip(
            istar.T,
            flip.T,
            home_s.T,
            *np.random.normal(size=(2, TRIALS, l_e)),
            *np.log(np.random.rand(2, TRIALS, l_e)),
        )
    ):
        if i % 1000 == 0:
            print(i)
        new_t0 = t0 + wt

        for e, (_s, _lt, _ts, _nt0) in enumerate(zip(s, index["l_t"], tq["t_s"], new_t0)):
            real_t = np.interp(_s, xp=np.arange(0.5, _lt), fp=_ts[:_lt])
            _np1 = lc(real_t - _nt0)
            if np.sum(_np1 - p1[e]) >= acct[e]:
                t0[e] = new_t0[e]
                p1[e] = _np1

            # 不设左右边界
            if NPE[e] == 0:
                step[e] = 1  # 只能创生
                accept[e] += np.log(4)  # 惩罚
            elif NPE[e] == 1 and step[e] == -1:
                # 1 -> 0: 行动后从 0 脱出的几率大，需要鼓励
                accept[e] -= np.log(4)

        t0_history[:, i] = t0
        e_create = step == 1

        loc[e_create] = periodic(home[e_create], index["l_t"][e_create])

        e_minus = ~e_create
        op = np.array(t * NPE, dtype=np.int32)
        loc[e_minus] = np.array([ _s[_op] for _s, _iop, _op in zip(s, e_minus, op) if _iop ], np.float32)

        e_move = step == 2
        e_pm = ~e_move
        step[e_move] = -1
        A_vec, c_vec = combine(A, cx, loc)
        Δν, beta = move1(A_vec, c_vec, z, step, index["mus"], index["sig2s"])

        ## move, step == 2
        # 待操作 PE 的新位置
        nloc = loc[e_move] + wander[e_move]
        nloc = periodic(nloc, index["l_t"][e_move])

        loc[e_move] = nloc
        Δcx_move, Δz_move = move2(A_vec[e_move], c_vec[e_move], -1, index["mus"][e_move], A[e_move], beta[e_move])
        A_vec1, c_vec1 = combine(A[e_move], cx[e_move] + Δcx_move, nloc)
        Δν1, beta1 = move1(A_vec1, c_vec1, z[e_move] + Δz_move, 1, index["mus"][e_move], index["sig2s"][e_move])
        Δν[e_move] += Δν1

        ## +1 cases, step == 2, 1
        e_plus = np.logical_or(e_move, e_create)
        real_t = np.array([ np.interp(_loc, xp=np.arange(0.5, _lt), fp=_ts[:_lt]) 
                            for _lt, _ts, _loc in zip(index["l_t"][e_plus], tq["t_s"][e_plus], loc[e_plus]) ])
        new_p1 = lc(real_t - t0[e_plus])
        Δν[e_plus] += new_p1

        ## -1 cases, step == 2, -1
        Δν[e_minus] -= np.array([ _p1[_op] for _p1, _iop, _op in zip(p1, e_minus, op) if _iop ], np.float32)

        ## non-move cases, step == 1, -1
        NPE[e_create] += 1
        Δν[e_pm] += step[e_pm] * (log_mu[e_pm] - np.log(tq["q_s"][e_pm, np.array(loc[e_pm], dtype=np.int32)]) - np.log(NPE[e_pm]))

        ## 计算 Δcx, Δz, 对 move-accept 进行特别处理
        step[e_move] = 1
        e_accept = Δν >= accept
        e_amove = np.logical_and(e_accept, e_move)
        ec_move = e_accept[e_move] # accept conditioned on moves
        beta[e_amove] = beta1[ec_move]
        A_vec[e_amove] = A_vec1[ec_move]
        c_vec[e_amove] = c_vec1[ec_move]
        Δcx, Δz = move2(A_vec[e_accept], c_vec[e_accept], step[e_accept], index["mus"][e_accept], A[e_accept], beta[e_accept])
        em_accept = e_move[e_accept] # moves conditioned on accept
        Δcx[em_accept] += Δcx_move[ec_move]
        Δz[em_accept] += Δz_move[ec_move]

        Δν[~e_accept] = 0
        step[~e_accept] = 0
        step[e_amove] = 2
        cx[e_accept] += Δcx
        z[e_accept] += Δz
        it_p1 = iter(new_p1)
        for e, (_s, _p1, _eacc, _ec, _em, _loc, _op) in enumerate(zip(s, p1, e_accept, e_create, e_move, loc, op)):
            if _eacc:
                if _ec: # 创生
                    _s.append(_loc)
                    p1[e] = np.append(p1[e], next(it_p1))
                elif _em: # 移动
                    _s[_op] = _loc
                    _p1[_op] = next(it_p1)
                else: # 消灭
                    del _s[_op]
                    p1[e] = np.delete(p1[e], _op)
        NPE = np.array([len(_s) for _s in s])

        Δν_history[:, i] = Δν
        flip[:, i] = step
        s0_history[:, i] = NPE
    return flip, Δν_history

l_e = len(index)
s_t = np.argsort(index["l_t"])
with h5py.File(fopt, "w") as opt:
    sample = opt.create_dataset("sample", shape=(l_e * TRIALS), 
                                dtype=[("TriggerNo", "u4"), ("ChannelID", "u4"),
                                       ("flip", "i2"), ("delta_nu", "f8")],
                                compression="gzip", shuffle=True, compression_opts=9)
    
    sample["TriggerNo"] = np.repeat(index["TriggerNo"], TRIALS)
    sample["ChannelID"] = np.repeat(index["ChannelID"], TRIALS)
    for part in range(l_e // args.size + 1):
        i_part = s_t[part * args.size:(part + 1) * args.size]
        if len(i_part):
            lp_t = np.max(index[i_part]["l_t"])
            lp_wave = np.max(index[i_part]["l_wave"])
            print(lp_t, lp_wave)
            flip, Δν_history = batch(A[i_part, :lp_wave, :lp_t], index[i_part], tq[i_part], z[i_part, :lp_wave])
            fi_part = (i_part * TRIALS)[:, None] + np.arange(TRIALS)[None, :]
            sample["flip"][fi_part.flatten()] = flip.flatten()
            sample["delta_nu"][fi_part.flatten()] = Δν_history.flatten()
