#!/usr/bin/env R
## 检查 compound poisson 的方差
##
## 用于理解光电子的电荷分布对能量分辨率的影响

library(plyr)

sigma_q <- 0.4
mu_q <- 1
N <- rpois(100000, lambda=4)
Q <- laply(N, function(N) {
    sum(rnorm(N, mean=mu_q, sd=sigma_q))
})
print(sprintf("数值模拟值：%.3f", var(Q) / var(N)))
print(sprintf("理论计算值：%.3f", (sigma_q^2 + 1)/mu_q))
