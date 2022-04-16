import argparse
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

psr = argparse.ArgumentParser()
psr.add_argument("ipt", type=str, nargs="+", help="input files")
psr.add_argument("-o", dest="opt", type=str, help="output file")
args = psr.parse_args()

samples = [
    pd.read_hdf(ipt, "sample").set_index(["TriggerNo", "ChannelID"]) for ipt in args.ipt
]


t0 = []
for sample in samples:
    t0.append(
        np.vstack(sample.groupby(level=[0, 1]).apply(lambda ent: ent["t0"].values)).T
    )

t0 = np.array(t0)

trials = range(100, 2500)
res = []
for i in trials:
    result = az.rhat(az.convert_to_dataset(t0[:, i : i * 2, :5]))
    res.append(np.array(result.x))

res = np.array(res)

plt.plot(np.array(trials) * 2, res)
plt.savefig(args.opt)
