SHELL:=bash
channelN:=$(shell seq -f '%02g' 0 29)
raw:=$(wildcard $(iptfold)/*.h5)
ifdef chunk
    raw:=$(iptfold)/$(chunk)x.h5
endif
charge:=$(patsubst $(iptfold)/%.h5,$(optfold)/%.h5,$(raw))
dist:=$(patsubst $(iptfold)/%.h5,$(method)/dist/%.h5,$(raw))
reco:=$(patsubst $(iptfold)/%.h5,$(method)/reco/%.csv,$(raw))
hist:=$(patsubst $(iptfold)/%.h5,$(method)/hist/%.pdf,$(raw))
ifeq ($(method), takara)
    predict:=nn
else
    predict:=fit
endif

PreData:=$(channelN:%=$(optfold)/CNN/PreProcess/Pre_Channel%.h5)
Nets:=$(channelN:%=$(optfold)/CNN/Nets/Channel%.torch_net)

.PHONY : sub

sub : $(charge)

test : $(hist) $(reco)
define fit
$(optfold)/%.h5 : $(iptfold)/%.h5 spe.h5
	@mkdir -p $$(dir $$@)
	export OMP_NUM_THREADS=2 && python3 fit.py $$< --met $(method) -N 50 --ref $$(wordlist 2,3,$$^) -o $$@ > $$@.log 2>&1
endef

define nn
$(optfold)/%.h5 : $(iptfold)/%.h5 spe.h5 $(Nets)
	@mkdir -p $$(dir $$@)
	python3 -u Prediction_Processing_Total.py $$< --met $(method) -o $$@ --ref $$(word 2,$$^) -N $(iptfold)/CNN/Nets -D 0 > $$@.log 2>&1
endef
$(eval $(call $(predict)))

$(method)/reco/%.csv : $(method)/dist/%.h5
	@mkdir -p $(dir $@)
	python3 csv_dist.py $^ -o $@
$(method)/hist/%.pdf : $(method)/dist/%.h5
	@mkdir -p $(dir $@)
	python3 draw_dist.py $^ -o $@
$(method)/dist/%.h5 : $(iptfold)/%.h5 $(optfold)/%.h5 spe.h5
	@mkdir -p $(dir $@)
	python3 test_dist.py $(word 2,$^) --ref $< $(word 3,$^) -o $@ > $@.log 2>&1

model : $(Nets)

$(optfold)/CNN/Nets/Channel%.torch_net : $(optfold)/CNN/Channel%/.Training_finished ;

$(optfold)/CNN/Channel%/.Training_finished : $(optfold)/CNN/PreProcess/Pre_Channel%.h5 spe.h5
	@mkdir -p $(dir $@)
	@mkdir -p $(optfold)/CNN
	python3 -u Data_Processing.py $< -n $* -B 64 --ref $(word $(words $^), $^) -o $(optfold)/CNN/Nets/Channel$*.torch_net $(dir $@) > $(dir $@)Train.log 2>&1
	@touch $@

$(PreData) : $(optfold)/CNN/.PreProcess

$(optfold)/CNN/.PreProcess : $(raw) spe.h5
	@mkdir -p $(optfold)/CNN/PreProcess
	python3 -u Data_Pre-Processing.py $(optfold)/ -o $(optfold)/CNN/PreProcess/Pre_Channel --ref $(word $(words $^), $^) > $(optfold)/CNN/PreProcess/PreProcess.log 2>&1
	@touch $@

$(iptfold)/$(chunk)x.h5 : $(iptfold)/$(chunk).h5
	python3 cut_data.py $^ -o $@ -a -1 -b 10000

spe.h5 : /mnt/stage/douwei/Simulation/1t_root/point_axis/1t_+0.000_x.h5
	python3 spe_get.py $^ -o $@ --num 500000 --len 80

model.pkl :
	python3 model.py $@

.DELETE_ON_ERROR: 

.SECONDARY:
