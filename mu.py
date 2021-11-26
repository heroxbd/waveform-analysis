import argparse
import pandas as pd
import numpy as np
import h5py
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, erf
from scipy.stats import norm
from numba import njit

psr = argparse.ArgumentParser()
psr.add_argument("-o", dest="opt", type=str, help="output file")
psr.add_argument("ipt", type=str, help="input file")
psr.add_argument("--ref", type=str, help="truth file")
args = psr.parse_args()

tc = ["TriggerNo", "ChannelID"]
sample = pd.read_hdf(args.ipt, "sample").set_index(tc)
s_history = pd.read_hdf(args.ipt, "s_history").set_index(tc)
mu0 = pd.read_hdf(args.ipt, "mu0").set_index(tc)
d_tlist = pd.read_hdf(args.ipt, "tlist").set_index(tc)

pe = pd.read_hdf(args.ref, "SimTriggerInfo/PEList").set_index(["TriggerNo", "PMTId"])
with h5py.File(args.ref) as ref:
    Tau = ref["Readout/Waveform"].attrs["tau"].item()
    Sigma = ref["Readout/Waveform"].attrs["sigma"].item()
    t0_truth = pd.DataFrame.from_records(ref["SimTruth/T"][:]).set_index(tc)


def lc(x, tau=Tau, sigma=Sigma):
    """
    light curve
    """
    if tau == 0.0:
        return norm.logpdf(x, loc=0, scale=sigma)
    else:
        alpha = 1 / tau
        co = -np.log(2.0 * tau) + alpha * alpha * sigma * sigma / 2.0
        x_erf = (alpha * sigma * sigma - x) / (np.sqrt(2.0) * sigma)
        return co + np.log(1.0 - erf(x_erf)) - alpha * x


Δt_r = Δt_l = 0
while lc(Δt_l) > np.log(1e-9):
    Δt_l -= 5
while lc(Δt_r) > np.log(1e-9):
    Δt_r += 5

pe_count = pe.groupby(level=[0, 1])["Charge"].count()


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


def rescale(ent):
    """
    find mu and t0.
    """
    eid, cid = ent.index[0]
    mu_t = mu0.loc[(eid, cid)][0]
    tlist = d_tlist.loc[(eid, cid)]
    t_s = tlist["t_s"].values
    ilp_cha = np.log(tlist["q_s"].sum()) - np.log(tlist["q_s"].values)

    N = len(t_s)  # N: number of t samples

    # throw away the burning stage
    size = np.uint32(len(ent))
    burn = size // 5
    es_history = s_history.loc[(eid, cid)]
    steps = es_history["step"].values
    second = np.searchsorted(steps, burn)
    if steps[second] == burn:
        first = second
    else:
        first = np.searchsorted(steps, steps[second - 1])
    steps[first:second] = burn
    es_history = es_history[first:]

    guess = ilp_cha[es_history["loc"].values.astype(int)]  # to multiply

    es_history["loc"] = np.interp(
        es_history["loc"].values, xp=np.arange(0.5, N), fp=t_s
    )
    t0_min = max(es_history["loc"].max() - Δt_r, t_s[0])
    t0_max = min(es_history["loc"].min() - Δt_l, t_s[-1])
    assert t0_min < t0_max, "interval is not found"

    rst = minimize_scalar(
        lambda x: -np.sum(lc(es_history["loc"].values - x)), bounds=(t0_min, t0_max)
    )
    if rst.success:
        t00 = rst.x
    else:
        t00 = 0.9 * t0_min + 0.1 * t0_max

    mu = mu_t

    def agg_NPE(t0):
        es_history["f"] = lc(es_history["loc"].values - t0) + guess
        return jit_agg_NPE(es_history["step"].values, es_history["f"].values, size)

    def t_t0(t0):
        nonlocal mu
        NPE, f_agg = agg_NPE(t0)
        rst = minimize_scalar(
            lambda μ: μ - logsumexp(NPE * np.log(μ / mu_t) + f_agg),
            bounds=(NPE[0], NPE[-1]),
        )

        if not rst.success:
            return np.inf
        else:
            mu = rst.x
            return rst.fun

    rst = minimize_scalar(t_t0, bounds=(t00 - 3, t00 + 3))
    assert rst.success, "t0 fit failed"
    t0 = rst.x

    return pd.Series(
        {
            "mu": mu,
            "t0": t0,
            "mu0": mu_t,
            "NPE_truth": pe_count.loc[(eid, cid)],
            "t0_truth": t0_truth.loc[(eid, cid)]["T0"],
        }
    )


mu_fit = sample.groupby(level=[0, 1]).apply(rescale)

with h5py.File(args.opt, "w") as opt:
    opt.create_dataset("mu", data=mu_fit.to_records(), compression="gzip", shuffle=True)
