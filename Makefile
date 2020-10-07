SHELL:=bash
jinpseq:=$(shell seq 5 8)
jinppre:=ztraining-
jinpchannelN:=$(shell seq -f '%02g' 0 29)
datfold:=/srv/waveform-analysis/dataset
tfold:=$(method)-$(mode)
prefix:=$($(set)pre)
seq:=$($(set)seq)
channelN:=$($(set)channelN)
ifdef chunk
    seq:=$(chunk)x
    datfoldi:=dataset
else
    datfoldi:=$(datfold)
endif
ifeq ($(method), takara)
    predict:=nn
else
    predict:=fit
endif

PreData:=$(channelN:%=$(datfold)/$(set)/PreProcess/Pre_Channel%.h5)
Nets:=$(channelN:%=$(datfold)/$(set)/$(mode)/Nets/Channel%.torch_net)

.PHONY : all

all : $(seq:%=$(tfold)/dist-$(set)/hist-%.pdf) $(seq:%=$(tfold)/dist-$(set)/record-%.csv)
$(tfold)/dist-$(set)/record-%.csv : $(tfold)/dist-$(set)/distr-%.h5
	python3 csv_dist.py $^ --mod $(mode) -o $@
$(tfold)/dist-$(set)/hist-%.pdf : $(tfold)/dist-$(set)/distr-%.h5
	python3 draw_dist.py $^ --mod $(mode) -o $@
$(tfold)/dist-$(set)/distr-%.h5 : $(datfoldi)/$(set)/$(prefix)%.h5 $(tfold)/resu-$(set)/sub-%.h5 spe-$(set).h5
	@mkdir -p $(dir $@)
	python3 test_dist.py $(word 2,$^) --mod $(mode) --ref $< $(word 3,$^) -o $@ > $@.log 2>&1
define fit
$(tfold)/resu-$(set)/sub-$(1).h5 : $(datfoldi)/$(set)/$(prefix)$(1).h5 spe-$(set).h5
	@mkdir -p $$(dir $$@)
	export OMP_NUM_THREADS=2 && python3 fit.py $$< --mod $(mode) --met $(method) --ref $$(wordlist 2,3,$$^) -o $$@ > $$@.log 2>&1
endef

define nn
$(tfold)/resu-$(set)/sub-$(1).h5 : $(datfoldi)/$(set)/$(prefix)$(1).h5 spe-$(set).h5 $(Nets)
	@mkdir -p $$(dir $$@)
	python3 -u Prediction_Processing_Total.py $$< --mod $(mode) --met $(method) -o $$@ --ref $$(word 2,$$^) -N $(datfold)/$(set)/$(mode)/Nets -D 0 > $$@.log 2>&1
endef
$(foreach i,$(seq),$(eval $(call $(predict),$(i))))

model : $(Nets)

$(datfold)/$(set)/$(mode)/Nets/Channel%.torch_net : $(datfold)/$(set)/$(mode)/$(prefix)Channel%/.Training_finished ;

$(datfold)/$(set)/$(mode)/$(prefix)Channel%/.Training_finished : $(datfold)/$(set)/PreProcess/Pre_Channel%.h5 spe-$(set).h5
	@mkdir -p $(dir $@)
	@mkdir -p $(datfold)/$(set)/$(mode)/Nets
	python3 -u Data_Processing.py $< -n $* -B 64 --mod $(mode) --ref $(word $(words $^), $^) -o $(datfold)/$(set)/$(mode)/Nets/Channel$*.torch_net $(dir $@) > $(dir $@)Train.log 2>&1
	@touch $@

$(PreData) : $(datfold)/$(set)/.PreProcess

$(datfold)/$(set)/.PreProcess : $(seq:%=$(datfold)/$(set)/$(prefix)%.h5) spe-$(set).h5
	@mkdir -p $(datfold)/$(set)/PreProcess
	python3 -u Data_Pre-Processing.py $(datfold)/$(set)/$(prefix) -o $(datfold)/$(set)/PreProcess/Pre_Channel --ref $(word $(words $^), $^) > $(datfold)/$(set)/PreProcess/PreProcess.log 2>&1
	@touch $@

$(datfoldi)/$(set)/$(prefix)$(chunk)x.h5 : $(datfold)/$(set)/$(prefix)$(chunk).h5
	@mkdir -p $(dir $@)
	python3 cut_data.py $^ -o $@ -a -1 -b 10000

spe-$(set).h5 : $(datfold)/$(set)/$(prefix)0.h5
	python3 spe_get.py $^ -o $@ --num 500000 --len 80

model.pkl :
	python3 model.py $@

.DELETE_ON_ERROR: 

.SECONDARY:
