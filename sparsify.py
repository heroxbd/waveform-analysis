import time
import argparse

import h5py as h5
import numpy as np
import cupy as cp
from scipy.signal import savgol_filter
from cupyx.scipy.signal import fftconvolve
from wf_func import clip, spe, read_model, Thres, likelihoodt0


def lucyddm(waveform, spe_pre, gmu):
    spe = cp.append(cp.zeros(len(spe_pre) - 1), cp.abs(spe_pre))
    waveform = cp.clip(waveform, 1e-6, cp.inf)
    spe = cp.clip(spe, 1e-6, cp.inf)[None, :]
    waveform /= gmu
    wave_deconv = waveform.copy()
    spe_mirror = spe[:, ::-1]
    for i in range(2000):
        relative_blur = waveform / fftconvolve(wave_deconv, spe, mode="same", axes=1)
        wave_deconv *= fftconvolve(relative_blur, spe_mirror, mode="same", axes=1)
    return wave_deconv


def initial_params(wave, char, spe_pre, gmu, Thres, p, nsp=4, nstd=3, is_t0=False, is_delta=False):
    hitt, char = clip(np.arange(len(wave)), char, Thres)
    char = char / char.sum() * np.clip(np.abs(wave.sum()), 1e-6, np.inf)
    tlist = np.unique(
        np.clip(np.hstack(hitt[:, None] + np.arange(-nsp, nsp + 1)), 0, len(wave) - 1)
    )

    index_prom = np.hstack(
        [
            np.argwhere(savgol_filter(wave, 11, 4) > nstd * spe_pre["std"]).flatten(),
            hitt,
        ]
    )
    left_wave = round(
        np.clip(index_prom.min() - 3 * spe_pre["mar_l"], 0, len(wave) - 1)
    )
    right_wave = round(
        np.clip(index_prom.max() + 3 * spe_pre["mar_r"], 0, len(wave) - 1)
    )
    wave = wave[left_wave:right_wave]

    npe_init = np.zeros(len(tlist))
    npe_init[np.isin(tlist, hitt)] = char / gmu
    tlist = np.sort(tlist)
    t_auto = np.arange(left_wave, right_wave)[:, None] - tlist
    A = spe((t_auto + np.abs(t_auto)) / 2, p[0], p[1], p[2])

    t0_init = None
    t0_init_delta = None
    if is_t0:
        t0_init, t0_init_delta = likelihoodt0(hitt=hitt, char=char, gmu=gmu, Tau=Tau, Sigma=Sigma, mode='charge', is_delta=is_delta)
    return A, wave, tlist, t0_init, t0_init_delta, npe_init, left_wave, right_wave

psr = argparse.ArgumentParser()
psr.add_argument("-o", dest="opt", type=str, help="output file")
psr.add_argument("ipt", type=str, help="input file")
psr.add_argument("--ref", type=str, help="reference file")
args = psr.parse_args()

fipt = args.ipt
fopt = args.opt
reference = args.ref

spe_pre = read_model(reference, 1)
with h5.File(fipt, "r", libver="latest", swmr=True) as ipt:
    ent = ipt["Readout/Waveform"][:]
    window = len(ent[0]["Waveform"])
    assert window >= len(spe_pre[0]["spe"]), "Single PE too long which is {}".format(
        len(spe_pre[0]["spe"])
    )
    Mu = ipt["Readout/Waveform"].attrs["mu"].item()
    Tau = ipt["Readout/Waveform"].attrs["tau"].item()
    Sigma = ipt["Readout/Waveform"].attrs["sigma"].item()
    gmu = ipt["SimTriggerInfo/PEList"].attrs["gmu"].item()
    gsigma = ipt["SimTriggerInfo/PEList"].attrs["gsigma"].item()

p = spe_pre[0]["parameters"]
std = 1.0
mix0sigma = 1e-3
mu0 = np.arange(1, int(Mu + 5 * np.sqrt(Mu)))
n_t = np.arange(1, 20)

ent = np.sort(ent, kind="stable", order=["TriggerNo", "ChannelID"])
Chnum = len(np.unique(ent["ChannelID"]))

d_tq = [("t_s", np.float32), ("q_s", np.float32), ("cq", np.float32)]

d_index = [
    ("loc", np.uint32),
    ("TriggerNo", np.uint32),
    ("ChannelID", np.uint32),
    ("mu0", np.float32),
    ("t0", np.float32),
    ("NPE", np.uint32),
    ("l_t", np.uint32),
    ("l_interval", np.uint32),
    ("a_wave", np.uint32),
    ("b_wave", np.uint32),
    ("l_wave", np.uint32),
    ("mus", np.float32),
    ("sig2s", np.float32),
    ("sig2w", np.float32),
    ("consumption", np.float32),
]

interval_l = []
t_l = []
q_l = []
cq_l = []  # cumulative q
a_w = []
b_w = []
mu_l = []
t0_l = []
NPE_l = []
mus_l = []
sig2s_l = []
sig2w_l = []
z_l = []
A_l = []
cx_l = []
s_l = []
consumption = []

cid = ent["ChannelID"]
assert np.all(cid == 0)
waves = ent["Waveform"].astype(np.float64) * spe_pre[0]["epulse"]
wave_deconv = lucyddm(cp.asarray(waves), cp.asarray(spe_pre[0]["spe"]), gmu)

n_wave = len(ent)  # number of waveforms

for ie, (e, wave, char) in enumerate(zip(ent, waves, cp.asnumpy(wave_deconv))):
    time_start = time.time()
    eid = e["TriggerNo"]
    cid = e["ChannelID"]

    A, y, tlist, t0_t, t0_delta, cha, left_wave, right_wave = initial_params(
        wave, char, spe_pre[cid], gmu, Thres["lucyddm"], p, is_t0=False
    )

    s_cha = np.cumsum(cha)
    # moving average filter of size 2*n+1
    cha = np.pad(s_cha[3:], (2, 1), "edge") - np.pad(s_cha[:-3], (2, 1), "edge")
    cha += 1e-8  # for completeness of the random walk.

    mu0 = abs(y.sum() / gmu)
    NPE = round(mu0)  # mu0: μ_total，LucyDDM 给出的 μ 猜测；NPE 是 PE 序列初值 s_0 的 PE 数。

    # Eq. (9) where the columns of A are taken to be unit-norm.
    mus = np.sqrt(np.diag(np.matmul(A.T, A)))  # ~39
    # elements of mus must be equal for continuous metropolis to function
    assert np.std(mus) < 1e-6, "mus must be equal"
    mus = mus[0]
    A = A / mus

    """
    A: basis dictionary
    p1: prior probability for each bin.
    sig2w: variance of white noise.
    sig2s: variance of signal x_i.
    mus: mean of signal x_i.
    TRIALS: number of Metropolis steps.
    """

    sig2w = spe_pre[cid]["std"] ** 2
    sig2s = (mus * (gsigma / gmu)) ** 2  # ~94

    # Only for multi-gaussian with arithmetic sequence of mu and sigma
    l_wave, l_t = A.shape

    # Eq. (29)
    cx = A / sig2w
    # mu = 0 => (y - A * mu -> z)
    z = y

    p_cha = cha / np.sum(cha)
    # q_s: pdf of LucyDDM charge (由 charge 引导的 PE 强度流先验)
    cq = np.cumsum(p_cha)
    cq[0] = 0  # 第一位须置0，否则会外溢
    cq[-1] = 1  # 最后置1，否则会外溢

    # t 的初始位置，取值为 {0.5, 1.5, 2.5, ..., (NPE-0.5)} / NPE 的 InverseCDF
    # MCMC 链的 PE configuration 初值 s0
    s = np.interp((np.arange(NPE) + 0.5) / NPE, xp=cq, fp=np.arange(l_t))

    # 直接算出 cx 和 z，防止大数相减积累误差。
    frac, ti = np.modf(s)
    ti = np.array(ti, np.int_)
    A_vec = (1 - frac)[None, :] * A[:, ti] + frac[None, :] * A[:, ti + 1]
    phi = A_vec @ A_vec.T * sig2s + np.eye(l_wave) * sig2w
    phi_inv = np.linalg.inv(phi)
    cx = phi_inv @ A
    z -= np.sum(A_vec, axis=1) * mus

    interval_l.append(tlist[-1] - tlist[0] + 1)
    t_l.append(tlist)
    q_l.append(p_cha)
    cq_l.append(cq)
    a_w.append(left_wave)
    b_w.append(right_wave)
    mu_l.append(mu0)
    t0_l.append(t0_t)
    NPE_l.append(NPE)
    mus_l.append(mus)
    sig2s_l.append(sig2s)
    sig2w_l.append(sig2w)
    z_l.append(z)
    A_l.append(A)
    cx_l.append(cx)
    s_l.append(s)
    consumption.append(time.time() - time_start)

A_shape = np.array([x.shape for x in A_l])
# lengths of waveform and tlist
l_wave, l_t = np.max(A_shape, axis=0)
l_interval = np.max(interval_l)
mNPE = np.max(NPE_l)

opts = {"compression": "gzip", "shuffle": True}

A = np.zeros((n_wave, l_wave, l_t), np.float32)
z = np.zeros((n_wave, l_wave), np.float32)
cx = np.zeros((n_wave, l_wave, l_t), np.float32)
s = np.zeros((n_wave, mNPE), np.float32)
tq = np.zeros((n_wave, l_t), d_tq)
t_index = np.zeros((n_wave, l_interval), np.uint32)

for loc, _lw, _lt, _npe, _t, _q, _cq, _z, _A, _cx, _s in zip(
    range(n_wave),
    A_shape[:, 0],
    A_shape[:, 1],
    NPE_l,
    t_l,
    q_l,
    cq_l,
    z_l,
    A_l,
    cx_l,
    s_l,
):
    tq["t_s"][loc, :_lt] = _t
    tq["q_s"][loc, :_lt] = _q
    tq["cq"][loc, :_lt] = _cq
    z[loc, :_lw] = _z
    A[loc, :_lw, :_lt] = _A
    cx[loc, :_lw, :_lt] = _cx
    s[loc, :_npe] = _s
    t_index[loc, _t - _t[0]] = 1

t_index = np.cumsum(t_index, axis=1, dtype=np.uint32) - 1

with h5.File(fopt, "w") as opt:
    opt.create_dataset("A", data=A, **opts)
    opt.create_dataset("cx", data=cx, **opts)
    opt.create_dataset("s", data=s, **opts)
    opt.create_dataset("tq", data=tq, **opts)
    opt.create_dataset("z", data=z, **opts)
    opt.create_dataset("t_index", data=t_index, **opts)
    index = opt.create_dataset("index", shape=(n_wave,), dtype=d_index, **opts)

    index["loc"] = np.arange(n_wave, dtype=np.uint32)
    index["TriggerNo"] = ent["TriggerNo"]
    index["ChannelID"] = ent["ChannelID"]
    index["mu0"] = mu_l
    index["t0"] = np.array(t0_l).flatten()
    index["NPE"] = NPE_l
    index["mus"] = mus_l
    index["sig2s"] = sig2s_l
    index["sig2w"] = sig2w_l
    index["a_wave"] = a_w
    index["b_wave"] = b_w
    index["l_wave"] = A_shape[:, 0]
    index["l_t"] = A_shape[:, 1]
    index["l_interval"] = interval_l
    index["consumption"] = consumption
print(f"Sparsify finished, real time {np.sum(consumption):.02f}s")
