import time
import argparse

import h5py
import numpy as np

# np.seterr(all='raise')
from scipy.special import erf
from scipy.stats import norm

import wf_func as wff

global_start = time.time()
cpu_global_start = time.process_time()

psr = argparse.ArgumentParser()
psr.add_argument("-o", dest="opt", type=str, help="output file")
psr.add_argument("ipt", type=str, help="input file")
psr.add_argument("--ref", type=str, help="reference file")
args = psr.parse_args()

fipt = args.ipt
fopt = args.opt
reference = args.ref

spe_pre = wff.read_model(reference, 1)
with h5py.File(fipt, "r", libver="latest", swmr=True) as ipt:
    ent = ipt["Readout/Waveform"][:100]
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
Thres = wff.Thres
mix0sigma = 1e-3
mu0 = np.arange(1, int(Mu + 5 * np.sqrt(Mu)))
n_t = np.arange(1, 20)

ent = np.sort(ent, kind="stable", order=["TriggerNo", "ChannelID"])
Chnum = len(np.unique(ent["ChannelID"]))

d_tq = [
    ("t_s", np.float32),
    ("q_s", np.float32),
]

d_index = [
    ("loc", np.uint32),
    ("TriggerNo", np.uint32),
    ("ChannelID", np.uint32),
    ("mu0", np.float32),
    ("l_t", np.uint32),
    ("l_wave", np.uint32),
    ("mus", np.float32),
    ("sig2s", np.float32),
    ("sig2w", np.float32),
]

t_l = []
q_l = []
mu_l = []
mus_l = []
sig2s_l = []
sig2w_l = []
z_l = []
A_l = []

n_wave = len(ent)  # number of waveforms

for ie, e in enumerate(ent):
    eid = e["TriggerNo"]
    cid = e["ChannelID"]
    assert cid == 0

    wave = e["Waveform"].astype(np.float64) * spe_pre[cid]["epulse"]

    # initialization
    A, y, tlist, t0_t, t0_delta, cha, left_wave, right_wave = wff.initial_params(
        wave,
        spe_pre[e["ChannelID"]],
        Tau,
        Sigma,
        gmu,
        Thres["lucyddm"],
        p,
        is_t0=False,
        is_delta=False,
        n=1,
    )
    s_cha = np.cumsum(cha)
    # moving average filter of size 2*n+1
    cha = np.pad(s_cha[3:], (2, 1), "edge") - np.pad(s_cha[:-3], (2, 1), "edge")
    cha += 1e-8  # for completeness of the random walk.

    mu_t = abs(y.sum() / gmu)
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
    # N: number of t bins
    # M: length of the waveform clip
    M, N = A.shape

    # Eq. (29)
    cx = A / sig2w
    # mu = 0 => (y - A * mu -> z)
    z = y

    p_cha = cha / np.sum(cha)

    t_l.append(tlist)
    q_l.append(p_cha)
    mu_l.append(mu_t)
    mus_l.append(mus)
    sig2s_l.append(sig2s)
    sig2w_l.append(sig2w)
    z_l.append(z)
    A_l.append(A)

A_shape = np.array([x.shape for x in A_l])
# lengths of waveform and tlist
l_wave, l_t = np.max(A_shape, axis=0)

with h5py.File(fopt, "w") as opt:
    A = opt.create_dataset(
        "A",
        shape=(n_wave, l_wave, l_t),
        dtype=np.float32,
        compression="gzip",
        shuffle=True,
        compression_opts=9,
    )
    tq = opt.create_dataset(
        "tq",
        shape=(n_wave, l_t),
        dtype=d_tq,
        compression="gzip",
        shuffle=True,
        compression_opts=9,
    )
    z = opt.create_dataset(
        "z",
        shape=(n_wave, l_wave),
        dtype=np.float32,
        compression="gzip",
        shuffle=True,
        compression_opts=9,
    )
    index = opt.create_dataset(
        "index",
        shape=(n_wave,),
        dtype=d_index,
        compression="gzip",
        shuffle=True,
        compression_opts=9,
    )

    index["loc"] = np.arange(n_wave, dtype=np.uint32)
    index["TriggerNo"] = ent["TriggerNo"]
    index["ChannelID"] = ent["ChannelID"]
    index["mu0"] = mu_l
    index["mus"] = mus_l
    index["sig2s"] = sig2s_l
    index["sig2w"] = sig2w_l
    index["l_wave"] = A_shape[:, 0]
    index["l_t"] = A_shape[:, 1]

    for loc, _lw, _lt, _t, _q, _z, _A in zip(
        range(n_wave), index["l_wave"], index["l_t"], t_l, q_l, z_l, A_l
    ):
        tq[loc, :_lt] = list(zip(_t, _q))
        z[loc, :_lw] = _z
        A[loc, :_lw, :_lt] = _A
