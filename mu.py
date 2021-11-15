import argparse
import pandas as pd
import numpy as np
import h5py
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', type=str, help='output file')
psr.add_argument('ipt', type=str, help='input file')
psr.add_argument('--ref', type=str, help='truth file')
args = psr.parse_args()

sample = pd.read_hdf(args.ipt, "sample").set_index(["TriggerNo", "ChannelID"])
mu0 = pd.read_hdf(args.ipt, "mu0").set_index(["TriggerNo", "ChannelID"])
pe = pd.read_hdf(args.ref, "SimTriggerInfo/PEList").set_index(["TriggerNo", "PMTId"])
pe_count = pe.groupby(level=[0, 1])['Charge'].count()

def rescale(ent):
    '''
    rescale mu0 to a better mu.
    '''
    eid, cid = ent.index[0]
    mu_t = mu0.loc[(eid, cid)][0]
    NPE0 = int(mu_t + 0.5)
    NPE_truth = pe_count.loc[(eid, cid)]
    size = len(ent)
    burn = size // 5
    steps = ent['flip'].values
    steps[np.abs(steps) == 2] = 0
    NPE_evo = np.cumsum(np.insert(steps, 0, NPE0))[burn:]
    NPE, counts = np.unique(NPE_evo, return_counts=True)

    freq = counts / (size - burn)
    loggN = -NPE * np.log(mu_t) + np.log(freq)

    is_bracket = True
    try:
        assert NPE[0] <= mu_t and mu_t <= NPE[-1], "not a bracket"
        rst = minimize_scalar(lambda μ: μ - logsumexp(loggN + NPE * np.log(μ)),
                              bracket=(NPE[0], mu_t, NPE[-1]))
    except (ValueError, AssertionError):
        is_bracket = False
        rst = minimize_scalar(lambda μ: μ - logsumexp(loggN + NPE * np.log(μ)),
                              bounds=(NPE[0], NPE[-1]))
    
    return pd.Series({'TriggerNo': eid, 
                      'ChannelID': cid,
                      'mu': rst.x,
                      'is_bracket': is_bracket,
                      'mu0': mu_t,
                      'NPE_truth': NPE_truth,
                      'NPE_mean': np.average(NPE_evo),
                      'NPE_median': np.median(NPE_evo),
                      })

mu_fit = sample.groupby(level=[0, 1]).apply(rescale)

with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('mu', data=mu_fit.to_records(index=False), 
                       compression="gzip", shuffle=True)
