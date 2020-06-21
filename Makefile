SHELL:=bash
jinpseq:=$(shell seq 0 9)
jinppre:=ztraining-
jinpchannelN:=$(shell seq -f '%02g' 0 29)
junoseq:=2 4
junopre:=junoWave
junochannelN:=$(shell seq 0 3)
datfold:=/srv/waveform-analysis/dataset
fragnum:=49
fragseq:=$(shell seq 0 ${fragnum})
tfold:=$(method)
ifdef chunk
	seq:=x
    datfoldi:=dataset
else
	seq:=$($(set)seq)
	channelN:=$($(set)channelN)
    datfoldi:=$(datfold)
endif
prefix:=$($(set)pre)
mod:=tot sub
ifeq ($(method), mcmc)
    core:=inference.py
else
    core:=fit.py
endif
ifeq ($(method), takara)
    predict:=nn
else
    predict:=fit
endif

PreData:=$(channelN:%=$(datfold)/$(set)/PreProcess/Pre_Channel%.h5)
Nets:=$(channelN:%=$(datfold)/$(set)/Nets/Channel%.torch_net)

.PHONY : all

all: $(rseq:%=$(tfold)/dist-$(set)/hist-sub-%.pdf) $(rseq:%=$(tfold)/dist-$(set)/record-sub-%.csv) $(rseq:%=$(tfold)/dist-$(set)/hist-tot-%.pdf) $(rseq:%=$(tfold)/dist-$(set)/record-tot-%.csv)
define measure
$(tfold)/dist-$(set)/record-$(1)-%.csv: $(tfold)/dist-$(set)/distr-$(1)-%.h5
	python3 csv_dist.py $$^ -o $$@
$(tfold)/dist-$(set)/hist-$(1)-%.pdf: $(tfold)/dist-$(set)/distr-$(1)-%.h5
	python3 draw_dist.py $$^ -o $$@
$(tfold)/dist-$(set)/distr-$(1)-%.h5: $(datfoldi)/$(set)/$(prefix)%.h5 $(tfold)/resu-$(set)/$(1)-%.h5 spe-$(set).h5
	@mkdir -p $$(dir $$@)
	python3 test_dist.py $$(word 2,$$^) --ref $$< $$(word 3,$$^) -o $$@ > $$@.log 2>&1
endef
$(foreach i,$(mod),$(eval $(call measure,$(i))))
$(tfold)/resu-$(set)/sub-%.h5: $(tfold)/resu-$(set)/tot-%.h5
	@mkdir -p $(dir $@)
	python3 adjust.py $^ -o $@
define fit
$(tfold)/resu-$(set)/tot-$(1).h5: $(fragseq:%=$(tfold)/unad-$(set)/unad-$(1)-%.h5)
	@mkdir -p $$(dir $$@)
	python3 integrate.py $$^ --num ${fragnum} --met $(method) -o $$@
$(tfold)/unad-$(set)/unad-$(1)-%.h5: $(datfoldi)/$(set)/$(prefix)$(1).h5 spe-$(set).h5
	@mkdir -p $$(dir $$@)
	export OMP_NUM_THREADS=2 && python3 $(core) $$< --met $(method) --ref $$(word 2,$$^) --num $(fragnum) -o $$@ > $$@.log 2>&1
endef
define nn
$(tfold)/resu-$(set)/tot-$(1).h5 : $(datfold)/$(set)/$(prefix)$(1).h5 $(Nets) | .Bulletin
	@mkdir -p $(dir $$@)
	python3 -u Prediction_Processing_Total.py $$< $$@ $(datfold)/$(set)/Nets -D 1 > $(dir $$@)Analysis.log 2>&1
endef
$(foreach i,$(rseq),$(eval $(call $(predict),$(i))))

.Bulletin:
	rm -f ./.bulletin.swp

model : $(Nets)

$(datfold)/$(set)/Nets/Channel%.torch_net : $(datfold)/$(set)/$(prefix)Channel%/.Training_finished ;

$(datfold)/$(set)/$(prefix)Channel%/.Training_finished : $(datfold)/$(set)/PreProcess/Pre_Channel%.h5 | .Bulletin
	@mkdir -p $(dir $@)
	@mkdir -p $(datfold)/$(set)/Nets
	python3 -u Data_Processing.py $^ -n $* -B 64 -o $(datfold)/$(set)/Nets/Channel$*.torch_net $(dir $@) > $(dir $@)Train.log 2>&1
	@touch $@

$(PreData) : PreProcess

PreProcess : $(seq:%=$(datfold)/$(set)/$(prefix)%.h5)
	@mkdir -p $(datfold)/$(set)/PreProcess
	python3 -u Data_Pre-Processing.py $(datfold)/$(set)/$(prefix) -o $(datfold)/$(set)/PreProcess/Pre_Channel -N $(seq) > $(datfold)/$(set)/PreProcess/PreProcess.log 2>&1

$(datfoldi)/$(set)/$(prefix)x.h5: $(datfold)/$(set)/$(prefix)$(chunk).h5
	@mkdir -p $(dir $@)
	python3 cut_data.py $^ -o $@ -a -1 -b 10000

spe-$(set).h5: $(datfold)/$(set)/$(prefix)$(word 1,$($(set)seq)).h5
	python3 spe_get.py $^ -o $@ --num 10000 --len 80

JUNO-Kaon-50.h5:
	wget http://hep.tsinghua.edu.cn/~orv/distfiles/JUNO-Kaon-50.h5

$(datfold)/$(set)/junoWave2.h5:
	@mkdir -p $(dir $@)
	wget https://cloud.tsinghua.edu.cn/f/f6e4cf503be542d3892f/?dl=1 -O $@
$(datfold)/$(set)/junoWave4.h5:
	@mkdir -p $(dir $@)
	wget https://cloud.tsinghua.edu.cn/f/846ecb6335564714902b/?dl=1 -O $@

.DELETE_ON_ERROR: 

.SECONDARY:
