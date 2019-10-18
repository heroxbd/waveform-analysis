outfile := $(patsubst %, result/%-pgan.h5, $(shell seq 1797))

all:medium result answer

answer:result/Total.h5 finalanswer.h5

result/Total.h5:result
	python3 alltran.py

finalanswer.h5:result/Total.h5
	python3 adjust.py

result:$(outfile)

medium:medium/average1.h5 medium/singlewave1.h5 medium/singlewave2.h5

medium/singlewave1.h5:
	python3 read.py 0
medium/average1.h5:medium/singlewave1.h5

medium/singlewave2.h5:
	python3 read.py 1
medium/average2.h5:medium/singlewave2.h5

result/%-pgan.h5:medium
	wolframscript -file finalfit.wl $(patsubst result/%-pgan.h5, %, $@)

clean:
	rm result/*-pgan.h5
