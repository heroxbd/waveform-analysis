SHELL:=bash
channelN:=$(shell seq -f '%02g' 0 0)
mu:=$(shell seq -f '%0.1f' 0.5 0.5 3.5 && seq -f '%0.1f' 4 2 10 && seq -f '%0.1f' 15 5 30)

tau:=$(shell awk -F',' 'NR == 1 { print $1 }' rc.csv)
sigma:=$(shell awk -F',' 'NR == 2 { print $1 }' rc.csv)

erg:=$(filter-out %-00-00,$(shell for i in $(mu); do for j in $(tau); do for k in $(sigma); do echo $${i}-$${j}-$${k}; done; done; done))
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
ifeq ($(method), fsmp)
	predict:=gibbs
	sparsify:=$(patsubst waveform/%.h5,result/$(method)/sparsify/%.h5,$(sim))
	# predict:=bayesian
else
    predict:=fit
endif
endif
endif

PreData:=$(channelN:%=result/takara/PreProcess/Pre_Channel%.h5)
Nets:=$(channelN:%=result/takara/char/Nets/Channel%.torch_net)

.PHONY : all

all : test

char : $(char)

test : $(hist)

solu : $(solu)

sparsify : $(sparsify)

sim : $(sim)

define gibbs
result/$(method)/sparsify/%.h5: waveform/%.h5 spe.h5
	@mkdir -p $$(dir $$@)
	./sel_GPU python3 sparsify.py $$< --ref $$(word 2,$$^) -o $$@ > $$@.log 2>&1

result/$(method)/batch/%.h5: result/$(method)/sparsify/%.h5 waveform/%.h5
	@mkdir -p $$(dir $$@)
	./sel_GPU python3 batch.py $$< --ref $$(word 2,$$^) -o $$@ --size 5000 > $$@.log 2>&1

result/$(method)/mu/%.h5: result/$(method)/batch/%.h5 result/$(method)/sparsify/%.h5 waveform/%.h5
	@mkdir -p $$(dir $$@)
	python3 mu.py $$< --sparse $$(word 2,$$^) --ref $$(word 3,$$^) -o $$@ > $$@.log 2>&1

result/$(method)/char/%.h5: result/$(method)/batch/%.h5 result/$(method)/mu/%.h5 result/$(method)/sparsify/%.h5 waveform/%.h5
	@mkdir -p $$(dir $$@)
	python3 collect.py $$< --mu $$(word 2,$$^) --sparse $$(word 3,$$^) --ref $$(word 4,$$^) -N 4 -o $$@ > $$@.log 2>&1
endef

define bayesian
result/$(method)/char/%.h5 : waveform/%.h5 spe.h5
	@mkdir -p $$(dir $$@)
	OMP_NUM_THREADS=2 python3 bayesian.py $$< --met $(method) -N 100 --ref $$(word 2,$$^) -o $$@ > $$@.log 2>&1
endef

define fit
result/$(method)/char/%.h5 : waveform/%.h5 spe.h5
	@mkdir -p $$(dir $$@)
	OMP_NUM_THREADS=2 python3 fit.py $$< --met $(method) -N 100 --ref $$(wordlist 2,3,$$^) -o $$@ > $$@.log 2>&1
endef

define nn
result/takara/char/%.h5 : waveform/%.h5 spe.h5 $(Nets)
	@mkdir -p $$(dir $$@)
	python3 -u predict.py $$< --met takara -o $$@ --ref $$(word 2,$$^) -N result/takara/char/Nets -D 0 > $$@.log 2>&1
endef
$(eval $(call $(predict)))

result/$(method)/dist/%.h5 : waveform/%.h5 result/$(method)/char/%.h5 spe.h5
	@mkdir -p $(dir $@)
	OMP_NUM_THREADS=2 python3 test_dist.py $(word 2,$^) --ref $< $(word 3,$^) -o $@ > $@.log 2>&1
result/$(method)/solu/%.h5 : result/$(method)/char/%.h5 waveform/%.h5 spe.h5
	@mkdir -p $(dir $@)
	OMP_NUM_THREADS=2 python3 toyRec.py $< --ref $(wordlist 2,3,$^) -o $@ > $@.log 2>&1
result/$(method)/hist/%.pdf : result/$(method)/dist/%.h5 waveform/%.h5 result/$(method)/solu/%.h5 result/$(method)/char/%.h5
	@mkdir -p $(dir $@)
	python3 draw_dist.py $< --ref $(wordlist 2,4,$^) -o $@ > $@.log 2>&1

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

define batches
batch/$(1)/%.h5: result/fsmp/sparsify/%.h5 waveform/%.h5
	mkdir -p $$(dir $$@)
	./sel_GPU python3 batch.py $$< --ref $$(word 2,$$^) -o $$@ --size 5000

endef

batch_id:=$(shell seq 0 4)

$(eval $(foreach i,$(batch_id),$(call batches,$(i))))

conv/%.pdf: $(foreach i,$(batch_id),batch/$(i)/%.h5)
	mkdir -p $(dir $@)
	python3 conv.py $^ -o $@

.DELETE_ON_ERROR: 

.SECONDARY:
