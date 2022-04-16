SHELL:=bash
channelN:=$(shell seq -f '%02g' 0 0)
mu:=$(shell seq -f '%0.1f' 0.5 0.5 3.5 && seq -f '%0.1f' 4 2 10 && seq -f '%0.1f' 15 5 30)

tau:=$(shell awk -F',' 'NR == 1 { print $1 }' rc.csv)
sigma:=$(shell awk -F',' 'NR == 2 { print $1 }' rc.csv)

config:=20-5

method:=fbmp
sim:=$(erg:%=waveform/%.h5)
char:=$(patsubst waveform/%.h5,result/$(method)/char/%.h5,$(sim))
solu:=$(patsubst waveform/%.h5,result/$(method)/solu/%.h5,$(sim))
dist:=$(patsubst waveform/%.h5,result/$(method)/dist/%.h5,$(sim))
hist:=$(patsubst waveform/%.h5,result/$(method)/hist/%.pdf,$(sim))
ifeq ($(method), takara)
    predict:=nn
else
ifeq ($(method), mcmc)
	predict:=bayesian
else
ifeq ($(method), fbmp)
	predict:=bayesian
else
    predict:=fit
endif
endif
endif

PreData:=$(channelN:%=result/takara/PreProcess/Pre_Channel%.h5)
Nets:=$(channelN:%=result/takara/char/Nets/Channel%.torch_net)

.PHONY : all

all : $(config:%=bias/%.pdf)

char : $(char)

test : $(hist)

solu : $(solu)

sim : $(sim)

sparsify/%.h5: waveform/%.h5 spe.h5
	mkdir -p $(dir $@)
	sem --fg python3 sparsify.py $< --ref $(word 2,$^) -o $@

batch/%.h5: sparsify/%.h5
	mkdir -p $(dir $@)
	sem --fg python3 batch.py $^ -o $@ --size 5000

batch2/%.h5: sparsify/%.h5
	mkdir -p $(dir $@)
	sem --fg python3 batch.py $^ -o $@ --size 5000

conv/%.pdf: batch/%.h5 batch2/%.h5
	mkdir -p $(dir $@)
	python3 conv.py $^ -o $@

tensorbatch/%.h5: sparsify/%.h5
	mkdir -p $(dir $@)
	sem --fg python3 tensorbatch.py $^ -o $@ --size 1000

mu/%.h5: batch/%.h5 sparsify/%.h5 waveform/%.h5 
	mkdir -p $(dir $@)
	python3 mu.py $< --sparse $(word 2,$^) --ref $(word 3,$^) -o $@

define bayesian
result/$(method)/char/%.h5 : waveform/%.h5 spe.h5
	@mkdir -p $$(dir $$@)
	OMP_NUM_THREADS=2 python3 bayesian.py $$< --met $(method) -N 1 --ref $$(word 2,$$^) -o $$@ > $$@.log 2>&1
endef

define fit
result/$(method)/char/%.h5 : waveform/%.h5 spe.h5
	@mkdir -p $$(dir $$@)
	OMP_NUM_THREADS=2 python3 fit.py $$< --met $(method) -N 1 --ref $$(wordlist 2,3,$$^) -o $$@ > $$@.log 2>&1
endef

define nn
result/takara/char/%.h5 : waveform/%.h5 spe.h5 $(Nets)
	@mkdir -p $$(dir $$@)
	python3 -u predict.py $$< --met takara -o $$@ --ref $$(word 2,$$^) -N result/takara/char/Nets -D 0 > $$@.log 2>&1
endef
$(eval $(call $(predict)))

result/$(method)/hist/%.pdf : result/$(method)/dist/%.h5 waveform/%.h5 result/$(method)/solu/%.h5 result/$(method)/char/%.h5
	@mkdir -p $(dir $@)
	python3 draw_dist.py $< --ref $(wordlist 2,4,$^) -o $@ > $@.log 2>&1
result/$(method)/dist/%.h5 : waveform/%.h5 result/$(method)/char/%.h5 spe.h5
	@mkdir -p $(dir $@)
	OMP_NUM_THREADS=2 python3 test_dist.py $(word 2,$^) --ref $< $(word 3,$^) -o $@ > $@.log 2>&1
result/$(method)/solu/%.h5 : result/$(method)/char/%.h5 waveform/%.h5 spe.h5
	@mkdir -p $(dir $@)
	OMP_NUM_THREADS=2 python3 toyRec.py $< --ref $(wordlist 2,3,$^) -o $@ > $@.log 2>&1

vs : rc.csv
	python3 vs.py --conf $^

model : $(Nets)

result/takara/char/Nets/Channel%.torch_net : result/takara/char/Channel%/.Training_finished ;

result/takara/char/Channel%/.Training_finished : result/takara/PreProcess/Pre_Channel%.h5 spe.h5
	@mkdir -p $(dir $@)
	@mkdir -p result/takara/char/Nets/
	python3 -u training.py $< -n $* -B 64 --ref $(word $(words $^), $^) -o result/takara/char/Nets/Channel$*.torch_net $(dir $@) > $(dir $@)Train.log 2>&1
	@touch $@

PreData : $(PreData)
$(PreData) : result/takara/.PreProcess

result/takara/.PreProcess : $(sim) spe.h5
	@mkdir -p result/takara/PreProcess
	python3 -u dataset.py waveform/ -o result/takara/PreProcess/Pre_Channel --ref $(word $(words $^), $^) > result/takara/PreProcess/PreProcess.log 2>&1
	@touch $@

waveform/%.h5 :
	@rm -f spe.h5
	@mkdir -p $(dir $@)
	python3 toySim.py --mts $* --noi -N 10000 -o $@ > $@.log 2>&1

spe.h5 : $(sim) ;

.SECONDEXPANSION:
bias/%.pdf: $$(foreach i,$(mu),mu/$$(i)-%.h5)
	mkdir -p $(dir $@)
	./bias.R $^ -o $@

.DELETE_ON_ERROR: 

.SECONDARY:
