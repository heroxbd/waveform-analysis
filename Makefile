SHELL:=bash
jinpseq:=$(shell seq 0 0)
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
	python3 test/csv_dist.py $$^ -o $$@
$(tfold)/dist-$(set)/hist-$(1)-%.pdf: $(tfold)/dist-$(set)/distr-$(1)-%.h5
	python3 test/draw_dist.py $$^ -o $$@
$(tfold)/dist-$(set)/distr-$(1)-%.h5: $(datfoldi)/$(set)/$(prefix)%.h5 $(tfold)/resu-$(set)/$(1)-%.h5 test/spe-$(set).h5
	@mkdir -p $$(dir $$@)
	python3 test/test_dist.py $$(word 2,$$^) --ref $$< $$(word 3,$$^) -o $$@ > $$@.log 2>&1
endef
$(foreach i,$(mod),$(eval $(call measure,$(i))))
define split
$(tfold)/resu-$(set)/sub-$(1).h5: $(tfold)/resu-$(set)/tot-$(1).h5
	@mkdir -p $$(dir $$@)
	python3 test/adjust.py $$^ -o $$@
$(tfold)/resu-$(set)/tot-$(1).h5: $(fragseq:%=$(tfold)/unad-$(set)/unad-$(1)-%.h5)
	@mkdir -p $$(dir $$@)
	python3 test/integrate.py $$^ --num ${fragnum} --met $(method) -o $$@
$(tfold)/unad-$(set)/unad-$(1)-%.h5: $(datfoldi)/$(set)/$(prefix)$(1).h5 test/spe-$(set).h5
	@mkdir -p $$(dir $$@)
	export OMP_NUM_THREADS=2 && python3 test/$(core) $$< --met $(method) --ref $$(word 2,$$^) --num $(fragnum) -o $$@ > $$@.log 2>&1
endef
$(foreach i,$(seq),$(eval $(call split,$(i))))

$(datfoldi)/$(set)/$(prefix)x.h5: $(datfold)/$(set)/$(prefix)$(chunk).h5
	@mkdir -p $(dir $@)
	python3 test/cut_data.py $^ -o $@ -a -1 -b 10000

test/spe-$(set).h5: $(datfold)/$(set)/$(prefix)$(word 1,$($(set)seq)).h5
	python3 test/spe_get.py $^ -o $@ --num 10000 --len 80

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
