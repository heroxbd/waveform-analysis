# waveform-analysis

## Methods:pill:

+ tara DL
+ xuyu DL
+ gz246 EMMP
+ wyy delta
+ xiaopeip
+ xdcFT
+ lucyddm
+ mcmc

## Frame:
The process of algorithm evaluation is automated in Makefile

For each method:
+ generate & save Answer of each training h5 file
+ record & save the efficiency of Answer generating
+ record & save average w&p-dist of each Answer respect to corresponding training h5 file

## Makefile argument:
+ set: jinp / juno
+ method: takara / xiaopeip / lucyddm / mcmc
+ mode: PEnum / Charge
+ rseq: the fileno to be extracted
+ chunk: the fileno for test
