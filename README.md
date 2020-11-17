# waveform-analysis

## Methods:

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
+ method: takara / xiaopeip / lucyddm
+ chunk: the fileno for test
+ iptfold: path of input h5 files
+ optfold: path of output h5 files
