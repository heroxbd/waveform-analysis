SHELL:=bash
jinpseq:=$(shell seq 0 9)
jinppre:=ztraining-
junoseq:=2 4
junopre:=junoWave
datfold:=/srv/waveform-analysis/dataset
fragnum:=49
fragseq:=$(shell seq 0 ${fragnum})
tfold:=test/$(method)
ifdef chunk
	seq:=x
    datfoldi:=dataset
else
	seq:=$($(set)seq)
    datfoldi:=$(datfold)
endif
prefix:=$($(set)pre)
mod:=tot sub
ifeq ($(method), mcmc)
    core:=inference.py
else
    core:=fit.py
endif

.PHONY: all

all: $(seq:%=$(tfold)/dist-$(set)/hist-sub-%.pdf) $(seq:%=$(tfold)/dist-$(set)/record-sub-%.csv) $(seq:%=$(tfold)/dist-$(set)/hist-tot-%.pdf) $(seq:%=$(tfold)/dist-$(set)/record-tot-%.csv)
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
define split
$(tfold)/resu-$(set)/sub-$(1).h5: $(tfold)/resu-$(set)/tot-$(1).h5
	@mkdir -p $$(dir $$@)
	python3 adjust.py $$^ -o $$@
$(tfold)/resu-$(set)/tot-$(1).h5: $(fragseq:%=$(tfold)/unad-$(set)/unad-$(1)-%.h5)
	@mkdir -p $$(dir $$@)
	python3 integrate.py $$^ --num ${fragnum} --met $(method) -o $$@
$(tfold)/unad-$(set)/unad-$(1)-%.h5: $(datfoldi)/$(set)/$(prefix)$(1).h5 spe-$(set).h5
	@mkdir -p $$(dir $$@)
	export OMP_NUM_THREADS=2 && python3 $(core) $$< --met $(method) --ref $$(word 2,$$^) --num $(fragnum) -o $$@ > $$@.log 2>&1
endef
$(foreach i,$(seq),$(eval $(call split,$(i))))

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

$(datfold)/$(set)/ztraining-0.h5:
$(datfold)/$(set)/ztraining-1.h5:
$(datfold)/$(set)/ztraining-2.h5:

.DELETE_ON_ERROR:
.SECONDARY:
all : help

help:
	@cat ./mannual.txt

jinppre:=ztraining-
junopre:=junoWave
jinpchannelN:=$(shell seq 0 29)
junochannelN:=$(shell seq 0 2)
jinpseq:=$(shell seq 0 9)
data_prefix:=$($(set)pre)
DataDir = /srv/waveform-analysis/dataset/$(set)
datfold:=/srv/waveform-analysis/dataset

TrainDataDir = $(DataDir)
NetDir = /srv/waveform-analysis/Network_Models_ztraining-all
PreDir = $(NetDir)/PreProcess
NetStore_prefix = $(NetDir)/$(data_prefix)Channel
TrainData = $(wildcard $(TrainDataDir)/$(data_prefix)*.h5)
PreData = $(shell seq -f "$(NetDir)/PreProcess/Pre_Channel%g.h5" 0 29)
Training = $(patsubst $(NetDir)/PreProcess/Pre_Channel%.h5, $(NetStore_prefix)%/.Training_finished, $(PreData))
Nets = $(patsubst $(NetDir)/PreProcess/Pre_Channel%.h5, $(NetDir)/Nets/Channel%.torch_net, $(PreData))
PreTrained_Model = 

ResultDir = ./
#test filename
testfile_prefix = $(DataDir)/$(data_prefix)
testfile = $(wildcard $(testfile_prefix)$(TestFileNo).h5)
# Result storage name
Result_prefix = $(ResultDir)Analyzing_Results_$(data_prefix)
Result = $(patsubst $(testfile_prefix)%.h5,$(Result_prefix)%/Prediction.h5, $(testfile))

test : $(patsubst $(testfile_prefix)%.h5,$(Result_prefix)%/DistanceDistribution.pdf,$(testfile)) $(patsubst $(testfile_prefix)%.h5,$(Result_prefix)%/DistanceDistribution.csv,$(testfile))

Analysis : $(Result)

model : $(Nets)

.Bulletin:
	rm -f ./.bulletin.swp

$(Result_prefix)%/Prediction.h5 : $(testfile_prefix)%.h5 $(Nets) | .Bulletin
	@mkdir -p $(dir $@)
	python3 -u Prediction_Processing_Total.py $< $@ $(NetDir)/Nets -D 1 > $(dir $@)Analysis.log 2>&1

$(Result_prefix)%/DistanceDistribution.csv : $(Result_prefix)%/Record.h5
	python3 -u ../csv_dist.py $^ -o $@

$(Result_prefix)%/DistanceDistribution.pdf : $(Result_prefix)%/Record.h5
	python3 -u ../draw_dist.py $^ -o $@

$(Result_prefix)%/Record.h5 : $(Result_prefix)%/Prediction.h5 $(testfile_prefix)%.h5
	python3 -u ../test_dist.py $< --ref $(word 2,$^) -o $@

$(PreData) : PreProcess

PreProcess : $(TrainData)
	@mkdir -p $(PreDir)
	python3 -u Data_Pre-Processing.py $(TrainDataDir)/$(data_prefix) -o $(PreDir) -N $(jinpseq) > $(PreDir)/PreProcess.log 2>&1

$(NetStore_prefix)%/.Training_finished : $(NetDir)/PreProcess/Pre_Channel%.h5 | .Bulletin
	@mkdir -p $(dir $@)
	python3 -u Data_Processing.py $^ -n $* -B 64 -o $(dir $@) -P $(PreTrained_Model) > $(dir $@)Train.log  2>&1 
	@touch $@

$(NetDir)/Nets/Channel%.torch_net : $(NetStore_prefix)%/.Training_finished
	@mkdir -p $(dir $@)
	python3 -u Choose_Nets.py $(dir $<) $@

clean :

.DELETE_ON_ERROR: 

.SECONDARY:

.PHONY : all test clean

.PHONY : Analysis model .Bulletin

