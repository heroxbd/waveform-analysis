# -*- coding: utf-8 -*-

import os
import pickle
import argparse
import pystan

psr = argparse.ArgumentParser()
psr.add_argument('ipt', type=str, help='input file')
args = psr.parse_args()

def main(Model_pkl):
    if not os.path.exists(Model_pkl):
        ocode = """
        data {
            int<lower=0> Nf;
            int<lower=0> Np;
            matrix[Nf, Np] m;
            vector[Nf] y;
        }
        parameters {
            vector<lower=0>[Np] x;
            real<lower=0> sigma;
        }
        model {
            y ~ normal_id_glm(m, 0, x, sigma);
        }
        """
        sm = pystan.StanModel(model_code=ocode)
        with open(Model_pkl, 'wb') as f:
            pickle.dump(sm, f)
    return

if __name__ == '__main__':
    main(args.ipt)
