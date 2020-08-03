SHELL:=bash
jinpseq:=$(shell seq 5 8)
jinppre:=ztraining-
jinpchannelN:=$(shell seq -f '%02g' 0 29)
junoseq:=2 4
junopre:=junoWave
junochannelN:=$(shell seq 0 3)
fragnum:=50
datfold:=/srv/waveform-analysis/dataset
tfold:=$(method)-$(mode)
prefix:=$($(set)pre)
seq:=$($(set)seq)
channelN:=$($(set)channelN)
ifdef chunk
    rseq:=$(chunk)x
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

all : $(rseq:%=$(tfold)/dist-$(set)/hist-%.pdf) $(rseq:%=$(tfold)/dist-$(set)/record-%.csv)
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
	export OMP_NUM_THREADS=2 && python3 fit.py $$< --mod $(mode) --met $(method) --ref $$(wordlist 2,3,$$^) -N $(fragnum) -o $$@ > $$@.log 2>&1
endef

define nn
$(tfold)/resu-$(set)/sub-$(1).h5 : $(datfoldi)/$(set)/$(prefix)$(1).h5 spe-$(set).h5 $(Nets)
	@mkdir -p $$(dir $$@)
	python3 -u Prediction_Processing_Total.py $$< --mod $(mode) --met $(method) -o $$@ --ref $$(word 2,$$^) -N $(datfold)/$(set)/$(mode)/Nets -D 1 > $$@.log 2>&1
endef
$(foreach i,$(rseq),$(eval $(call $(predict),$(i))))

model : $(Nets)

$(datfold)/$(set)/$(mode)/Nets/Channel%.torch_net : $(datfold)/$(set)/$(mode)/$(prefix)Channel%/.Training_finished ;

$(datfold)/$(set)/$(mode)/$(prefix)Channel%/.Training_finished : $(datfold)/$(set)/PreProcess/Pre_Channel%.h5
	@mkdir -p $(dir $@)
	@mkdir -p $(datfold)/$(set)/$(mode)/Nets
	python3 -u Data_Processing.py $^ -n $* -B 64 --mod $(mode) -o $(datfold)/$(set)/$(mode)/Nets/Channel$*.torch_net $(dir $@) > $(dir $@)Train.log 2>&1
	@touch $@

$(PreData) : PreProcess

PreProcess : $(seq:%=$(datfold)/$(set)/$(prefix)%.h5) spe-$(set).h5
	@mkdir -p $(datfold)/$(set)/PreProcess
	python3 -u Data_Pre-Processing.py $(datfold)/$(set)/$(prefix) -o $(datfold)/$(set)/PreProcess/Pre_Channel --ref $(word $(words $^), $^) > $(datfold)/$(set)/PreProcess/PreProcess.log 2>&1

$(datfoldi)/$(set)/$(prefix)$(chunk)x.h5 : $(datfold)/$(set)/$(prefix)$(chunk).h5
	@mkdir -p $(dir $@)
	python3 cut_data.py $^ -o $@ -a -1 -b 100000

spe-$(set).h5 : $(datfold)/$(set)/$(prefix)0.h5
	python3 spe_get.py $^ -o $@ --num 500000 --len 80

model.pkl :
	python3 model.py $@

JUNO-Kaon-50.h5 :
	wget http://hep.tsinghua.edu.cn/~orv/distfiles/JUNO-Kaon-50.h5

$(datfold)/$(set)/junoWave2.h5 :
	@mkdir -p $(dir $@)
	wget https://cloud.tsinghua.edu.cn/f/f6e4cf503be542d3892f/?dl=1 -O $@
$(datfold)/$(set)/junoWave4.h5 :
	@mkdir -p $(dir $@)
	wget https://cloud.tsinghua.edu.cn/f/846ecb6335564714902b/?dl=1 -O $@

.DELETE_ON_ERROR: 

.SECONDARY:
