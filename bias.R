#!/usr/bin/env Rscript

library(argparse)
library(rhdf5)
library(plyr)
library(ggplot2)
library(reshape2)
psr <- ArgumentParser(description='Plot bias distributions')
psr$add_argument("ipt", help="input", nargs="+")
psr$add_argument("-o", dest="opt", help="output")
args <- psr$parse_args()

loadh5s <- function(fn) { 
    mu <- h5read(fn, "mu")
    intensity <- as.numeric(strsplit(basename(fn), '-')[[1]][1])
    i_truth <- intensity / (1 - exp(-intensity))
    data.frame(NPE=mu$NPE_truth, intensity=i_truth, metropolis=mu$mu, charge=mu$mu0, 
               config=as.factor(intensity), t0=mu$t0, t0_truth=mu$t0_truth)
}
mu <- ldply(args$ipt, loadh5s)

mu_long <- melt(mu, measure.vars=c("metropolis", "charge"), id.vars=c("NPE", "intensity", "config"))
mu_long$rbias <- (mu_long$value - mu_long$NPE) / mu_long$NPE
mu_long$irbias <- (mu_long$value - mu_long$intensity) / mu_long$intensity
mu_long$bias <- mu_long$value - mu_long$NPE
mu_long$NPE <- as.factor(mu_long$NPE)

mu$NPE <- as.factor(mu$NPE)

pdf(args$opt, 4, 4)
p0 <- ggplot(mu_long, aes(x=NPE , y=rbias)) + geom_boxplot() + facet_wrap(~variable)
p0 <- p0 + ylab("Relative Bias")
p1 <- p0 + aes(y=bias)
print(p0)
print(p0 %+% mu_long[as.numeric(mu_long$NPE) <= 10, ])
print(p0 + aes(x=config))

e0 <- ggplot(mu_long, aes(x=config, y=irbias, color=variable)) + ylab("Relative Bias")
e0 <- e0 + geom_pointrange(stat="summary",
                          position=position_dodge(width=0.2),
                          fun=function(x) {mean(x)},
                          fun.min=function(x) {mean(x)-sd(x)/sqrt(length(x))},
                          fun.max=function(x) {mean(x)+sd(x)/sqrt(length(x))})
print(e0)
print(p1)
print(p1 %+% mu_long[as.numeric(mu_long$NPE) <= 10, ])

time0 <- ggplot(mu, aes(x=NPE, y=t0 - t0_truth)) + ylab("Time Bias/ns")
print(time0 + geom_boxplot())
time1 <- time0 + aes(x=config) + geom_pointrange(stat="summary",
                                                fun=function(x) {mean(x)},
                                                fun.min=function(x) {mean(x)-sd(x)/sqrt(length(x))},
                                                fun.max=function(x) {mean(x)+sd(x)/sqrt(length(x))})
print(time1)

dev.off()
