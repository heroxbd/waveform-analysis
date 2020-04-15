SHELL:=bash
set:=jinp
waveseq:=$(shell seq 0 1)
#set:=juno
#waveseq:=2 4
Dir:=dataset
fragnum:=0
fragseq:=$(shell seq 0 ${fragnum})
xdcFTp:=test/xdcFT
xiaoPp:=test/xiaopeip
lucy:=test/lucyddm
mcmc:=test/mcmc

.PHONY: all
all: xdcFT xiaopeip lucyddm mcmc junoDataset

mcmc: $(waveseq:%=$(mcmc)/dist-$(set)/hist-%.pdf) $(mcmc)/dist-$(set)/record.csv
$(mcmc)/dist-$(set)/record.csv: $(waveseq:%=$(mcmc)/dist-$(set)/record-%.csv)
	cat $^ > $@
$(mcmc)/dist-$(set)/record-%.csv: $(mcmc)/dist-$(set)/distrecord-%.h5
	python3 test/csv_dist.py $^ -o $@
$(mcmc)/dist-$(set)/hist-%.pdf: $(mcmc)/dist-$(set)/distrecord-%.h5
	python3 test/draw_dist.py $^ -o $@
$(mcmc)/dist-$(set)/distrecord-%.h5: $(Dir)/$(set)/*%.h5 $(mcmc)/sub-$(set)/submission-%.h5
	mkdir -p $(dir $@)
	python3 test/test_dist.py $(word 2,$^) --ref $< -o $@ > $@.log 2>&1
define mcmc_split
$(mcmc)/sub-$(set)/submission-$(1).h5: $(fragseq:%=$(mcmc)/unad-$(set)/unadjusted-$(1)-%.h5)
	mkdir -p $$(dir $$@)
	python3 test/integrate.py $$^ --num ${fragnum} -o $$@
$(mcmc)/unad-$(set)/unadjusted-$(1)-%.h5: $(Dir)/$(set)/*$(1).h5 test/spe-$(set).h5
	mkdir -p $$(dir $$@)
	python3 $(mcmc)/mcmcfit.py $$< --ref $$(word 2,$$^) --num ${fragnum} -o $$@
endef
$(foreach i,$(waveseq),$(eval $(call mcmc_split,$(i))))

lucyddm: $(waveseq:%=$(lucy)/dist-$(set)/hist-%.pdf) $(lucy)/dist-$(set)/record.csv
$(lucy)/dist-$(set)/record.csv: $(waveseq:%=$(lucy)/dist-$(set)/record-%.csv)
	cat $^ > $@
$(lucy)/dist-$(set)/record-%.csv: $(lucy)/dist-$(set)/distrecord-%.h5
	python3 test/csv_dist.py $^ -o $@
$(lucy)/dist-$(set)/hist-%.pdf: $(lucy)/dist-$(set)/distrecord-%.h5
	python3 test/draw_dist.py $^ -o $@
$(lucy)/dist-$(set)/distrecord-%.h5: $(Dir)/$(set)/*%.h5 $(lucy)/sub-$(set)/submission-%.h5
	mkdir -p $(dir $@)
	python3 test/test_dist.py $(word 2,$^) --ref $< -o $@
$(lucy)/sub-$(set)/submission-%.h5 : $(Dir)/$(set)/*%.h5 test/spe-$(set).h5
	mkdir -p $(dir $@)
	python3 $(lucy)/lucyDDM.py $< --ref $(word 2,$^) -o $@

xiaopeip: $(waveseq:%=$(xiaoPp)/dist-$(set)/hist-%.pdf) $(xiaoPp)/dist-$(set)/record.csv
$(xiaoPp)/dist-$(set)/record.csv: $(waveseq:%=$(xiaoPp)/dist-$(set)/record-%.csv)
	cat $^ > $@
$(xiaoPp)/dist-$(set)/record-%.csv: $(xiaoPp)/dist-$(set)/distrecord-%.h5
	python3 test/csv_dist.py $^ -o $@
$(xiaoPp)/dist-$(set)/hist-%.pdf: $(xiaoPp)/dist-$(set)/distrecord-%.h5
	python3 test/draw_dist.py $^ -o $@
$(xiaoPp)/dist-$(set)/distrecord-%.h5: $(Dir)/$(set)/*%.h5 $(xiaoPp)/sub-$(set)/submission-%.h5
	mkdir -p $(dir $@)
	python3 test/test_dist.py $(word 2,$^) --ref $< -o $@ > $@.log 2>&1
define xpp_split
$(xiaoPp)/sub-$(set)/submission-$(1).h5: $(xiaoPp)/submission/total-$(1).h5
	python3 $(xiaoPp)/adjust.py $$^ -o $$@
$(xiaoPp)/sub-$(set)/total-$(1).h5: $(fragseq:%=$(xiaoPp)/unad-$(set)/unadjusted-$(1)-%.h5)
	mkdir -p $$(dir $$@)
	python3 test/integrate.py $$^ --num ${fragnum} -o $$@
$(xiaoPp)/unad-$(set)/unadjusted-$(1)-%.h5: $(Dir)/$(set)/*$(1).h5 test/spe-$(set).h5
	mkdir -p $$(dir $$@)
	python3 $(xiaoPp)/finalfit.py $$< --ref $$(word 2,$$^) --num ${fragnum} -o $$@
endef
$(foreach i,$(waveseq),$(eval $(call xpp_split,$(i))))

xdcFT: $(waveseq:%=$(xdcFTp)/dist-$(set)/hist-%.pdf) $(xdcFTp)/dist-$(set)/record.csv
$(xdcFTp)/dist-$(set)/record.csv: $(waveseq:%=$(xdcFTp)/dist-$(set)/record-%.csv)
	cat $^ > $@
$(xdcFTp)/dist-$(set)/record-%.csv: $(xdcFTp)/dist-$(set)/distrecord-%.h5
	python3 test/csv_dist.py $^ -o $@
$(xdcFTp)/dist-$(set)/hist-%.pdf: $(xdcFTp)/dist-$(set)/distrecord-%.h5
	python3 test/draw_dist.py $^ -o $@
$(xdcFTp)/dist-$(set)/distrecord-%.h5: $(Dir)/$(set)/*%.h5 $(xdcFTp)/sub-$(set)/submission-%.h5
	mkdir -p $(dir $@)
	python3 test/test_dist.py $(word 2,$^) --ref $< -o $@
$(xdcFTp)/sub-$(set)/submission-%.h5 : $(Dir)/$(set)/*%.h5 test/spe-$(set).h5
	mkdir -p $(dir $@)
	python3 $(xdcFTp)/FFT_decon.py $< --ref $(word 2,$^) -o $@ -k 0.05 -a 4 -e 4

test/spe-$(set).h5: $(Dir)/$(set)/*$(word 1,$(waveseq)).h5
	python3 test/spe_get.py $^ -o $@ --num 10000 --len 80

JUNO-Kaon-50.h5:
	wget http://hep.tsinghua.edu.cn/~orv/distfiles/JUNO-Kaon-50.h5

junoDataset: $(junowaveseq:%=$(junoDir)/junoWave%.h5)
$(junoDir)/junoWave2.h5:
	mkdir -p $(dir $@)
	wget https://cloud.tsinghua.edu.cn/f/f6e4cf503be542d3892f/?dl=1 -O $@
$(junoDir)/junoWave4.h5:
	mkdir -p $(dir $@)
	wget https://cloud.tsinghua.edu.cn/f/846ecb6335564714902b/?dl=1 -O $@

.DELETE_ON_ERROR:
.SECONDARY:
