import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 14
plt.switch_backend("pgf")

bench_dtype = [("n", np.int_), ("t", np.float_)]

keys = ["a100", "2080", "2060", "k80", "r7", "6700", "cpu"]
descriptions = {
    "2060": r"NVIDIA\textsuperscript{\textregistered} RTX 2060 Max-Q",
    "2080": r"NVIDIA\textsuperscript{\textregistered} RTX 2080 Ti",
    "k80": r"NVIDIA\textsuperscript{\textregistered} Tesla K80",
    "a100": r"NVIDIA\textsuperscript{\textregistered} A100",
    "6700": r"AMD Radeon\texttrademark RX 6700 XT",
    "r7": r"AMD Radeon\texttrademark \uppercase\expandafter{\romannumeral7}",
    "cpu": r"AMD EPYC\texttrademark 7702 (CPU)",
}
bench = {}
for key in keys:
    bench[key] = np.loadtxt(
        "bench.{}.csv".format(key), dtype=bench_dtype, delimiter=","
    )

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(1, 1, 1)

for key in keys:
    data = bench[key][1:]
    ax.plot(data["n"], data["t"], label=descriptions[key])

ax.set_xlabel("number of waveforms")
ax.set_ylabel("time/s")
ax.set_title("FSMP performance")
ax.legend(bbox_to_anchor=(1, 1))

fig.savefig("bench_gpu.pdf", transparent=True, bbox_inches="tight")
fig.savefig("bench_gpu.pgf", transparent=True, bbox_inches="tight")
plt.close(fig)
