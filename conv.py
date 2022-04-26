import argparse
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

psr = argparse.ArgumentParser()
psr.add_argument("ipt", type=str, nargs="+", help="input files")
psr.add_argument("-o", dest="opt", type=str, help="output file")
args = psr.parse_args()

samples = [
    pd.read_hdf(ipt, "sample")
    .set_index(["TriggerNo", "ChannelID"])
    .groupby(level=[0, 1])
    for ipt in args.ipt
]


t0 = []
s0 = []
for sample in samples:
    t0.append(np.vstack(sample.apply(lambda ent: ent["t0"].values)).T)
    s0.append(np.vstack(sample.apply(lambda ent: ent["s0"].values)).T)

t0 = np.array(t0)
s0 = np.array(s0)
# chain, draw, x_dim_0
t0 = az.convert_to_dataset(t0)
s0 = az.convert_to_dataset(s0)

trials = range(100, 10000)
res_t0 = []
res_s0 = []
for i in trials:
    result = az.rhat(t0.sel(draw=slice(i, i * 2), x_dim_0=slice(None, 10)))
    res_t0.append(np.array(result.x))

    result = az.rhat(s0.sel(draw=slice(i, i * 2), x_dim_0=slice(None, 10)))
    res_s0.append(np.array(result.x))

res_t0 = np.array(res_t0)
res_s0 = np.array(res_s0)

with PdfPages(args.opt) as pp:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.array(trials) * 2, res_t0, label=[str(i) for i in range(11)])
    ax.set_xlabel("step")
    ax.set_ylabel("\hat{R}")
    ax.set_title("Convergence of t0")
    ax.legend()
    pp.savefig(fig)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.array(trials) * 2, res_s0, label=[str(i) for i in range(11)])
    ax.set_xlabel("step")
    ax.set_ylabel("\hat{R}")
    ax.set_title("Convergence of ||s||0")
    ax.legend()
    pp.savefig(fig)
