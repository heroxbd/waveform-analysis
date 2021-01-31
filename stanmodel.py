# -*- coding: utf-8 -*-

import os
import pickle
import pystan

ocode = """
functions {
    real mnormal_lpdf(real x, real pl, real s, real sigma) {
        return log((1 - pl) * exp(normal_lpdf(x | 0, s)) + pl * exp(normal_lpdf(x | 1, sigma)));
    }
}
data {
    int<lower=0> N;
    real<lower=0> Tau;
    real<lower=0> Sigma;
    real<lower=0> mu;
    real<lower=0> s;
    real<lower=0> sigma;
    real<lower=0> std;
    matrix[N, N] AV;
    vector[N] w;
}
transformed data {
    vector[N] t;
    real<lower=0> Alpha;
    real<lower=0> Co;
    for (n in 1:N) {
        t[n] = n - 1;
    }
    if (Tau != 0)
        Alpha = 1. / Tau;
        Co = Alpha / 2. * exp(Alpha * Alpha * Sigma * Sigma / 2.);
}
parameters {
    real t0;
    vector[N] A;
}
transformed parameters {
    vector[N] td;
    vector<lower=0>[N] pl;
    td = t - t0;
    if (Tau != 0)
        for (n in 1:N) {
            if (Tau != 0)
                pl[n] = Co * (1. - erf((Alpha * Sigma * Sigma - td[n]) / (sqrt2() * Sigma))) * exp(-Alpha * td[n]) * mu;
            else
                pl[n] = exp(normal_lpdf(td | 0, Sigma)) * mu;
        }
}
model {
    t0 ~ uniform(0, 600);
    for (n in 1:N) {
        A[n] ~ mnormal(pl[n], s, sigma);
    }
    w ~ normal(AV * A, std);
}
generated quantities {
    real Mu;
    Mu = sum(A);
}
"""
sm = pystan.StanModel(model_name='mix01', model_code=ocode)
with open('stanmodel.pkl', 'wb') as f:
    pickle.dump(sm, f)