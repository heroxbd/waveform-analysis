SHELL:=bash
channelN:=$(shell seq -f '%02g' 0 29)
raw:=$(wildcard $(datfold)/*.h5)
charge:=$(patsubst $(datfold)/%.h5,$(datfold)/charge_%.h5,$(raw))
dist:=$(patsubst $(datfold)/%.h5,$(method)/dist_%.h5,$(raw))
record:=$(patsubst $(datfold)/%.h5,$(method)/record_%.csv,$(raw))
hist:=$(patsubst $(datfold)/%.h5,$(method)/hist_%.pdf,$(raw))

ifdef chunk
    raw:=$(datfoldi)/$(chunk)x.h5
    datfoldi:=dataset
else
    datfoldi:=$(datfold)
endif
ifeq ($(method), takara)
    predict:=nn
else
    predict:=fit
endif

PreData:=$(channelN:%=$(datfold)/CNN/PreProcess/Pre_Channel%.h5)
Nets:=$(channelN:%=$(datfold)/CNN/Nets/Channel%.torch_net)

.PHONY : sub

sub : $(charge)

test : $(hist) $(record)
define fit
$(datfoldi)/charge_%.h5 : $(datfoldi)/%.h5 spe.h5
	export OMP_NUM_THREADS=2 && python3 fit.py $$< --met $(method) --ref $$(wordlist 2,3,$$^) -o $$@ > $$@.log 2>&1
endef

define nn
$(datfoldi)/charge_%.h5 : $(datfoldi)/%.h5 spe.h5 $(Nets)
	python3 -u Prediction_Processing_Total.py $$< --met $(method) -o $$@ --ref $$(word 2,$$^) -N $(datfold)/CNN/Nets -D 0 > $$@.log 2>&1
endef
$(eval $(call $(predict)))

$(method)/record_%.csv : $(method)/dist_%.h5
	python3 csv_dist.py $^ -o $@
$(method)/hist_%.pdf : $(method)/dist_%.h5
	python3 draw_dist.py $^ -o $@
$(method)/dist_%.h5 : $(datfoldi)/%.h5 $(datfoldi)/charge_%.h5 spe.h5
	@mkdir -p $(dir $@)
	python3 test_dist.py $(word 2,$^) --ref $< $(word 3,$^) -o $@ > $@.log 2>&1

model : $(Nets)

$(datfold)/CNN/Nets/Channel%.torch_net : $(datfold)/CNN/Channel%/.Training_finished ;

$(datfold)/CNN/Channel%/.Training_finished : $(datfold)/CNN/PreProcess/Pre_Channel%.h5 spe.h5
	@mkdir -p $(dir $@)
	@mkdir -p $(datfold)/CNN
	python3 -u Data_Processing.py $< -n $* -B 64 --ref $(word $(words $^), $^) -o $(datfold)/CNN/Nets/Channel$*.torch_net $(dir $@) > $(dir $@)Train.log 2>&1
	@touch $@

$(PreData) : $(datfold)/CNN/.PreProcess

$(datfold)/CNN/.PreProcess : $(raw) spe.h5
	@mkdir -p $(datfold)/CNN/PreProcess
	python3 -u Data_Pre-Processing.py $(datfold)/ -o $(datfold)/CNN/PreProcess/Pre_Channel --ref $(word $(words $^), $^) > $(datfold)/CNN/PreProcess/PreProcess.log 2>&1
	@touch $@

$(datfoldi)/$(chunk)x.h5 : $(datfold)/$(chunk).h5
	@mkdir -p $(dir $@)
	python3 cut_data.py $^ -o $@ -a -1 -b 10000

spe.h5 : /srv/waveform-analysis/ztraining-0.h5
	python3 spe_get.py $^ -o $@ --num 500000 --len 80

model.pkl :
	python3 model.py $@

.DELETE_ON_ERROR: 

.SECONDARY:
