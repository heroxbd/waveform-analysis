import argparse
import pandas as pd
import numpy as np
import h5py
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
import wf_func as wff

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', type=str, help='output file')
psr.add_argument('ipt', type=str, help='input file')
psr.add_argument('--ref', type=str, help='truth file')
args = psr.parse_args()

tc = ["TriggerNo", "ChannelID"]
sample = pd.read_hdf(args.ipt, "sample").set_index(tc)
s_history = pd.read_hdf(args.ipt, "s_history").set_index(tc)
mu0 = pd.read_hdf(args.ipt, "mu0").set_index(tc)
d_tlist = pd.read_hdf(args.ipt, "tlist").set_index(tc)

pe = pd.read_hdf(args.ref, "SimTriggerInfo/PEList").set_index(["TriggerNo", "PMTId"])
with h5py.File(args.ref) as ref:
    Tau = ref['Readout/Waveform'].attrs['tau'].item()
    Sigma = ref['Readout/Waveform'].attrs['sigma'].item()
    t0_truth = ref['SimTruth/T']['T0'][:]

def lc(Δt):
    '''
    light curve
    '''
    return wff.convolve_exp_norm(Δt, Tau, Sigma)
Δt_r = Δt_l = 0
while lc(Δt_l) > 1e-9:
    Δt_l -= 5
while lc(Δt_r) > 1e-9:
    Δt_r += 5

pe_count = pe.groupby(level=[0, 1])['Charge'].count()

def rescale(ent):
    '''
    find mu and t0.
    '''
    eid, cid = ent.index[0]
    mu_t = mu0.loc[(eid, cid)][0]
    tlist = d_tlist.loc[(eid, cid)]
    t_s = tlist['t_s'].values
    ilp_cha = np.log(tlist['q_s'].sum()) - np.log(tlist['q_s'].values)

    N = len(t_s) # N: number of t samples

    # throw away the burning stage
    size = np.uint32(len(ent))
    burn = size // 5
    es_history = s_history.loc[(eid, cid)]
    steps = es_history['step'].values
    second = np.searchsorted(steps, burn)
    if steps[second] == burn:
        first = second
    else:
        first = np.searchsorted(steps, steps[second - 1])
    steps[first:second] = burn
    es_history = es_history[first:]

    guess = ilp_cha[es_history['loc'].values.astype(int)] # to multiply

    es_history['loc'] = np.interp(es_history['loc'].values, xp = np.arange(0.5, N), fp = t_s)
    t0_min = max(es_history['loc'].max() - Δt_r, t_s[0])
    t0_max = min(es_history['loc'].min() - Δt_l, t_s[-1])
    assert t0_min < t0_max, "interval is not found"

    t0_prev = t0 = t0_min * 0.9 + t0_max * 0.1
    mu_prev = np.inf
    mu = mu_t
    def agg_NPE(t0):
        es_history['f'] = np.log(lc(es_history['loc'].values - t0)) + guess

        f_vec = es_history.groupby("step").agg(
            NPE = pd.NamedAgg('f', 'count'),
            f_vec = pd.NamedAgg('f', logsumexp)
        )
        f_vec['repeat'] = np.diff(np.append(f_vec.index.values, np.uint32(size)))

        NPE_vec = f_vec.groupby("NPE").apply(
            lambda x: logsumexp(x.f_vec, b=x.repeat))
        return NPE_vec.index.values, NPE_vec.values

    while np.abs(mu - mu_prev) / mu > 1e-5 or np.abs(t0_prev - t0) > 1e-5:
        NPE, f_agg = agg_NPE(t0)
        rst = minimize_scalar(
            lambda μ: μ - logsumexp(NPE * np.log(μ/mu_t) + f_agg),
            bounds=(NPE[0], NPE[-1]))

        assert rst.success, "mu fit failed"
        mu_prev = mu
        mu = rst.x * 0.9 + mu * 0.1
    
        expo = NPE * np.log(mu/mu_t)
        def t_t0(t0):
            NPE, f_agg = agg_NPE(t0)
            return - logsumexp(expo + f_agg)

        rst = minimize_scalar(t_t0, bounds=(t0_min, t0_max))
        assert rst.success, "t0 fit failed"
        t0_prev = t0
        t0 = rst.x * 0.9 + t0 * 0.1
        print(eid, mu, t0)

    print(t0, t0_truth[eid])
    print("-----")

    return pd.Series({'mu': mu,
                      't0': t0,
                      'mu0': mu_t,
                      'NPE_truth': pe_count.loc[(eid, cid)],
                      't0_truth': t0_truth[eid]
                      })

mu_fit = sample.groupby(level=[0, 1]).apply(rescale)

with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('mu', data=mu_fit.to_records(index=False), 
                       compression="gzip", shuffle=True)
