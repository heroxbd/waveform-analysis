mu:=$(shell seq -f '%0.1f' 0.5 0.5 3.5 && seq -f '%0.1f' 4 2 10 && seq -f '%0.1f' 15 5 25)

tau:=$(shell awk -F',' 'NR == 1 { print $1 }' ../rc.csv)
sigma:=$(shell awk -F',' 'NR == 2 { print $1 }' ../rc.csv)

all: elbo.png wdist.png

%.png:
	python3 ../compareELBO.py --idir ../result/fbmp ../result/fbmpZ -o $@ --tau $(tau) --sigma $(sigma) --type  nopri --describe xdc zaq --mus $(mu) --field $*
