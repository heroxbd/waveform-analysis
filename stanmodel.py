# -*- coding: utf-8 -*-

import os
import pickle
import pystan

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
    y ~ normal(m * x, sigma);
}
"""
sm = pystan.StanModel(model_code=ocode)
with open('stanmodel.pkl', 'wb') as f:
    pickle.dump(sm, f)