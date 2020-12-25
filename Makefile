SHELL:=bash
channelN:=$(shell seq -f '%02g' 0 0)
mu:=$(shell seq -f '%02g' 5 5 15)
method=lucyddm
raw:=$(mu:%=waveform/mu%.h5)
char:=$(patsubst waveform/%.h5,result/$(method)/char/%.h5,$(raw))
solu:=$(patsubst waveform/%.h5,result/$(method)/solu/%.h5,$(raw))
dist:=$(patsubst waveform/%.h5,result/$(method)/dist/%.h5,$(raw))
reco:=$(patsubst waveform/%.h5,result/$(method)/reco/%.csv,$(raw))
hist:=$(patsubst waveform/%.h5,result/$(method)/hist/%.pdf,$(raw))
ifeq ($(method), takara)
    predict:=nn
else
    predict:=fit
endif

PreData:=$(channelN:%=result/$(method)/char/PreProcess/Pre_Channel%.h5)
Nets:=$(channelN:%=result/$(method)/char/Nets/Channel%.torch_net)

.PHONY : all

all : sub test sim

sub : $(char) $(solu)

test : $(hist) $(reco)

sim : $(raw)

define fit
result/$(method)/char/%.h5 : waveform/%.h5 spe.h5
	@mkdir -p $$(dir $$@)
	OMP_NUM_THREADS=2 python3 fit.py $$< --met $(method) -N 50 --ref $$(wordlist 2,3,$$^) -o $$@ > $$@.log 2>&1
endef

define nn
result/$(method)/char/%.h5 : waveform/%.h5 spe.h5 $(Nets)
	@mkdir -p $$(dir $$@)
	python3 -u Prediction_Processing_Total.py $$< --met $(method) -o $$@ --ref $$(word 2,$$^) -N result/$(method)/char/Nets -D 0 > $$@.log 2>&1
endef
$(eval $(call $(predict)))

result/$(method)/solu/%.h5 : result/$(method)/char/%.h5 waveform/%.h5
	@mkdir -p $(dir $@)
	python3 toyRec.py $< --ref $(word 2,$^) --tau 40 --sigma 6 -o $@ > $@.log 2>&1

result/$(method)/reco/%.csv : result/$(method)/dist/%.h5 waveform/%.h5 result/$(method)/solu/%.h5
	@mkdir -p $(dir $@)
	python3 csv_dist.py $< --ref $(wordlist 2,3,$^) -o $@ > $@.log 2>&1
result/$(method)/hist/%.pdf : result/$(method)/dist/%.h5 waveform/%.h5 result/$(method)/solu/%.h5
	@mkdir -p $(dir $@)
	python3 draw_dist.py $< --ref $(wordlist 2,3,$^) -o $@ > $@.log 2>&1
result/$(method)/dist/%.h5 : waveform/%.h5 result/$(method)/char/%.h5 spe.h5
	@mkdir -p $(dir $@)
	python3 test_dist.py $(word 2,$^) --ref $< $(word 3,$^) -o $@ > $@.log 2>&1

model : $(Nets)

result/$(method)/char/Nets/Channel%.torch_net : result/$(method)/char/Channel%/.Training_finished ;

result/$(method)/char/Channel%/.Training_finished : result/$(method)/char/PreProcess/Pre_Channel%.h5 spe.h5
	@mkdir -p $(dir $@)
	@mkdir -p result/$(method)/char/Nets/
	python3 -u Data_Processing.py $< -n $* -B 64 --ref $(word $(words $^), $^) -o result/$(method)/char/Nets/Channel$*.torch_net $(dir $@) > $(dir $@)Train.log 2>&1
	@touch $@

$(PreData) : result/$(method)/char/.PreProcess

result/$(method)/char/.PreProcess : $(raw) spe.h5
	@mkdir -p result/$(method)/char/PreProcess
	python3 -u Data_Pre-Processing.py waveform/ -o result/$(method)/char/PreProcess/Pre_Channel --ref $(word $(words $^), $^) > result/$(method)/char/PreProcess/PreProcess.log 2>&1
	@touch $@

waveform/mu%.h5 :
	@rm -f spe.h5
	@mkdir -p $(dir $@)
	python3 toySim.py --mu $* --tau 40 --sigma 6 --noi -N 100000 -o $@ > $@.log 2>&1

spe.h5 : sim ;

clean :
	pushd waveform; rm -r ./* ; popd
	pushd result; rm -r ./* ; popd

model.pkl :
	python3 model.py $@

.DELETE_ON_ERROR: 

.SECONDARY: