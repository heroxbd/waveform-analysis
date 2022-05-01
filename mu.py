import argparse
import pandas as pd
import numpy as np
import h5py

psr = argparse.ArgumentParser()
psr.add_argument("-o", dest="opt", type=str, help="output file")
psr.add_argument("ipt", type=str, help="input file")
psr.add_argument("--sparse", type=str, help="LucyDDM file")
psr.add_argument("--ref", type=str, help="truth file")
args = psr.parse_args()

sample = pd.read_hdf(args.ipt, "sample").set_index(["TriggerNo", "ChannelID"])
index = pd.read_hdf(args.sparse, "index").set_index(["TriggerNo", "ChannelID"])
mu0 = index["mu0"]
pe = pd.read_hdf(args.ref, "SimTriggerInfo/PEList").set_index(["TriggerNo", "PMTId"])
pe_count = pe.groupby(level=[0, 1])["Charge"].count()
t0_truth = pd.read_hdf(args.ref, "SimTruth/T").set_index(["TriggerNo", "ChannelID"])

def rescale(ent):
    """
    rescale mu0 to a better mu.
    """
    eid, cid = ent.index[0]
    mu_t = mu0.loc[(eid, cid)]
    NPE0 = int(mu_t + 0.5)
    NPE_truth = pe_count.loc[(eid, cid)]
    size = len(ent)
    burn = size // 2
    steps = ent["flip"].values
    steps[np.abs(steps) == 2] = 0
    NPE_evo = np.cumsum(np.insert(steps, 0, NPE0))[burn:]

    is_bracket = True
    
    mu = np.average(ent["mu"].values[burn:])

    t0 = np.average(ent["t0"].values[burn:])

    return pd.Series(
        {
            "mu": mu,
            "is_bracket": is_bracket,
            "mu0": mu_t,
            "NPE_truth": NPE_truth,
            "NPE_mean": np.average(NPE_evo),
            "NPE_median": np.median(NPE_evo),
            "t0": t0,
            "t0_truth": t0_truth.loc[(eid, cid)]["T0"],
        }
    )


mu_fit = sample.groupby(level=[0, 1]).apply(rescale)

with h5py.File(args.opt, "w") as opt:
    opt.create_dataset("mu", data=mu_fit.to_records(), compression="gzip", shuffle=True)
