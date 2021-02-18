SHELL:=bash
channelN:=$(shell seq -f '%02g' 0 0)
mu:=$(shell seq -f '%02g' 1 1 5 && seq -f '%02g' 6 2 10 && seq -f '%02g' 15 5 30)
# mu:=$(shell seq -f '%02g' 15 5 15)

tau:=$(shell awk -F',' 'NR == 1 { print $1 }' rc.csv)
sigma:=$(shell awk -F',' 'NR == 2 { print $1 }' rc.csv)

erg:=$(filter-out %-00-00,$(shell for i in $(mu); do for j in $(tau); do for k in $(sigma); do echo $${i}-$${j}-$${k}; done; done; done))
# erg:=05-40-10
method=lucyddm
sim:=$(erg:%=waveform/%.h5)
char:=$(patsubst waveform/%.h5,result/$(method)/char/%.h5,$(sim))
solu:=$(patsubst waveform/%.h5,result/$(method)/solu/%.h5,$(sim))
dist:=$(patsubst waveform/%.h5,result/$(method)/dist/%.h5,$(sim))
hist:=$(patsubst waveform/%.h5,result/$(method)/hist/%.pdf,$(sim))
ifeq ($(method), takara)
    predict:=nn
else
ifeq ($(method), mcmcrec)
	predict:=mcmcrec
else
    predict:=fit
endif
endif

PreData:=$(channelN:%=result/takara/PreProcess/Pre_Channel%.h5)
Nets:=$(channelN:%=result/takara/char/Nets/Channel%.torch_net)

.PHONY : all

all : test

test : $(hist)

solu : $(solu)

sim : $(sim)

define mcmcrec
result/$(method)/char/%.h5 : waveform/%.h5 spe.h5
	@mkdir -p $$(dir $$@)
	python3 toyRecMCMC.py $$< --met $(method) -N 100 --ref $$(word 2,$$^) -o $$@ > $$@.log 2>&1
endef

define fit
result/$(method)/char/%.h5 : waveform/%.h5 spe.h5
	@mkdir -p $$(dir $$@)
	OMP_NUM_THREADS=2 python3 fit.py $$< --met $(method) -N 100 --ref $$(wordlist 2,3,$$^) -o $$@ > $$@.log 2>&1
endef

define nn
result/takara/char/%.h5 : waveform/%.h5 spe.h5 $(Nets)
	@mkdir -p $$(dir $$@)
	python3 -u Prediction_Processing_Total.py $$< --met takara -o $$@ --ref $$(word 2,$$^) -N result/takara/char/Nets -D 0 > $$@.log 2>&1
endef
$(eval $(call $(predict)))

result/$(method)/hist/%.pdf : result/$(method)/dist/%.h5 waveform/%.h5 result/$(method)/solu/%.h5
	@mkdir -p $(dir $@)
	python3 draw_dist.py $< --ref $(wordlist 2,3,$^) -o $@ > $@.log 2>&1
result/$(method)/dist/%.h5 : waveform/%.h5 result/$(method)/char/%.h5 spe.h5
	@mkdir -p $(dir $@)
	python3 test_dist.py $(word 2,$^) --ref $< $(word 3,$^) -o $@ > $@.log 2>&1
result/$(method)/solu/%.h5 : result/$(method)/char/%.h5 waveform/%.h5
	@mkdir -p $(dir $@)
	python3 toyRec.py $< --ref $(word 2,$^) -o $@ > $@.log 2>&1

vs : rc.csv
	python3 vs.py --conf $^

model : $(Nets)

result/takara/char/Nets/Channel%.torch_net : result/takara/char/Channel%/.Training_finished ;

result/takara/char/Channel%/.Training_finished : result/takara/PreProcess/Pre_Channel%.h5 spe.h5
	@mkdir -p $(dir $@)
	@mkdir -p result/takara/char/Nets/
	python3 -u Data_Processing.py $< -n $* -B 64 --ref $(word $(words $^), $^) -o result/takara/char/Nets/Channel$*.torch_net $(dir $@) > $(dir $@)Train.log 2>&1
	@touch $@

PreData : $(PreData)
$(PreData) : result/takara/.PreProcess

result/takara/.PreProcess : $(sim) spe.h5
	@mkdir -p result/takara/PreProcess
	python3 -u Data_Pre-Processing.py waveform/ -o result/takara/PreProcess/Pre_Channel --ref $(word $(words $^), $^) > result/takara/PreProcess/PreProcess.log 2>&1
	@touch $@

waveform/%.h5 :
	@rm -f spe.h5
	@mkdir -p $(dir $@)
	python3 toySim.py --mts $* --noi -N 1000 -o $@ > $@.log 2>&1

spe.h5 : $(sim) ;

.DELETE_ON_ERROR: 

.SECONDARY: