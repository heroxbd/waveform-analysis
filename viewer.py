#!/usr/bin/env python3

from matplotlib import pyplot as plt
import h5py
import argparse
import numpy as np

psr = argparse.ArgumentParser()
psr.add_argument('ipt', type=str, nargs="+", help='input files, including waveform, metropolis and mu')
psr.add_argument('--eid', type=int, help='event id')

args = psr.parse_args()

def entry(table, eid):
    loc0, loc1 = np.searchsorted(table['TriggerNo'], (args.eid, args.eid+1))
    return table[loc0:loc1]

with h5py.File(args.ipt[0]) as f_wave:
    wave = entry(f_wave['/Readout/Waveform'], args.eid)[0]['Waveform']
    pelist = entry(f_wave['/SimTriggerInfo/PEList'], args.eid)

with h5py.File(args.ipt[1]) as f_sample:
    sample = entry(f_sample['sample'], args.eid)
    tlist = entry(f_sample['tlist'], args.eid)
    history = entry(f_sample['s_history'], args.eid)
