#!/usr/bin/env python3

import argparse
psr = argparse.ArgumentParser()
psr.add_argument('ipt', help='input file')
psr.add_argument('opt', help='output symbolic link to net')
args = psr.parse_args()
output = args.opt
inputdir = args.ipt

import os
import re


fileSet = os.listdir(inputdir)
matchrule = re.compile(r"_epoch(\d+)_loss(\d+(\.\d*)?|\.\d+)([eE]([-+]?\d+))?")
NetLoss_reciprocal = []
for filename in fileSet :
    if "_epoch" in filename : NetLoss_reciprocal.append(1 / float(matchrule.match(filename)[2]))
    else : NetLoss_reciprocal.append(0)
net_name = fileSet[NetLoss_reciprocal.index(max(NetLoss_reciprocal))]

os.system("ln -s " + inputdir + "/" + net_name + " " + output)
