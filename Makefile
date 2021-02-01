SHELL:=bash
channelN:=$(shell seq -f '%02g' 0 0)
mu:=$(shell seq -f '%02g' 1 1 5 && seq -f '%02g' 6 2 10 && seq -f '%02g' 15 5 30)
tau:=$(shell seq -f '%02g' 0 20 100)
sigma:=$(shell seq -f '%02g' 0 5 15)

tau:=$(shell awk -F',' 'NR == 1 { print $1 }' rc.csv)
sigma:=$(shell awk -F',' 'NR == 2 { print $1 }' rc.csv)

erg:=$(filter-out %-00-00,$(shell for i in $(mu); do for j in $(tau); do for k in $(sigma); do echo $${i}-$${j}-$${k}; done; done; done))
# erg:=05-40-10
method=lucyddm
sim:=$(erg:%=waveform/%.h5)
char:=$(patsubst waveform/%.h5,result/$(method)/char/%.h5,$(sim))
solu:=$(patsubst waveform/%.h5,result/$(method)/solu/%.h5,$(sim))
dist:=$(patsubst waveform/%.h5,result/$(method)/dist/%.h5,$(sim))
reco:=$(patsubst waveform/%.h5,result/$(method)/reco/%.csv,$(sim))
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

PreData:=$(channelN:%=result/$(method)/PreProcess/Pre_Channel%.h5)
Nets:=$(channelN:%=result/$(method)/char/Nets/Channel%.torch_net)

.PHONY : all

all : solu vs

vs : result/$(method)/solu/vs.pdf

test : $(hist) $(reco)

solu : $(solu)

sim : $(sim)

define mcmcrec
result/$(method)/solu/%.h5 result/$(method)/char/%.h5 &: waveform/%.h5 spe.h5
	@mkdir -p $$(dir result/$(method)/solu/$$*.h5)
	@mkdir -p $$(dir result/$(method)/char/$$*.h5)
	python3 toyRecMCMC.py $$< --met $(method) -N 100 --ref $$(word 2,$$^) -o result/$(method)/solu/$$*.h5 result/$(method)/char/$$*.h5 > result/$(method)/solu/$$*.h5.log 2>&1
endef

define fit
result/$(method)/char/%.h5 : waveform/%.h5 spe.h5
	@mkdir -p $$(dir $$@)
	OMP_NUM_THREADS=2 python3 fit.py $$< --met $(method) -N 100 --ref $$(wordlist 2,3,$$^) -o $$@ > $$@.log 2>&1
endef

define nn
result/$(method)/char/%.h5 : waveform/%.h5 spe.h5 $(Nets)
	@mkdir -p $$(dir $$@)
	python3 -u Prediction_Processing_Total.py $$< --met $(method) -o $$@ --ref $$(word 2,$$^) -N result/$(method)/char/Nets -D 0 > $$@.log 2>&1
endef
$(eval $(call $(predict)))

result/$(method)/reco/%.csv : result/$(method)/dist/%.h5 waveform/%.h5 result/$(method)/solu/%.h5
	@mkdir -p $(dir $@)
	python3 csv_dist.py $< --ref $(wordlist 2,3,$^) -o $@ > $@.log 2>&1
result/$(method)/hist/%.pdf : result/$(method)/dist/%.h5 waveform/%.h5 result/$(method)/solu/%.h5
	@mkdir -p $(dir $@)
	python3 draw_dist.py $< --ref $(wordlist 2,3,$^) -o $@ > $@.log 2>&1
result/$(method)/dist/%.h5 : waveform/%.h5 result/$(method)/char/%.h5 spe.h5
	@mkdir -p $(dir $@)
	python3 test_dist.py $(word 2,$^) --ref $< $(word 3,$^) -o $@ > $@.log 2>&1
result/$(method)/solu/%.h5 : result/$(method)/char/%.h5 waveform/%.h5
	@mkdir -p $(dir $@)
	python3 toyRec.py $< --ref $(word 2,$^) -o $@ > $@.log 2>&1
result/$(method)/solu/vs.pdf : $(solu) rc.csv
	@mkdir -p $(dir $@)
	python3 vs.py --folder result/$(method)/solu waveform --conf $(word $(words $^), $^) -o $@ > $@.log 2>&1

model : $(Nets)

result/$(method)/char/Nets/Channel%.torch_net : result/$(method)/char/Channel%/.Training_finished ;

result/$(method)/char/Channel%/.Training_finished : result/$(method)/PreProcess/Pre_Channel%.h5 spe.h5
	@mkdir -p $(dir $@)
	@mkdir -p result/$(method)/char/Nets/
	python3 -u Data_Processing.py $< -n $* -B 64 --ref $(word $(words $^), $^) -o result/$(method)/char/Nets/Channel$*.torch_net $(dir $@) > $(dir $@)Train.log 2>&1
	@touch $@

$(PreData) : result/$(method)/.PreProcess

result/$(method)/.PreProcess : $(sim) spe.h5
	@mkdir -p result/$(method)/PreProcess
	python3 -u Data_Pre-Processing.py waveform/ -o result/$(method)/PreProcess/Pre_Channel --ref $(word $(words $^), $^) > result/$(method)/PreProcess/PreProcess.log 2>&1
	@touch $@

waveform/%.h5 :
	@rm -f spe.h5
	@mkdir -p $(dir $@)
	python3 toySim.py --mts $* --noi -N 10000 -o $@ > $@.log 2>&1

spe.h5 : $(sim) ;

clean :
	pushd waveform; rm -r ./* ; popd
	pushd result/$(method); rm -r ./* ; popd

model.pkl :
	python3 model.py $@

.DELETE_ON_ERROR: 

.SECONDARY: