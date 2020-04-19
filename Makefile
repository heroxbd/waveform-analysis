SHELL:=bash
jinpseq:=$(shell seq 0 0)
jinppre:=ztraining-
junoseq:=2 4
junopre:=junoWave
datfold:=dataset
fragnum:=99
fragseq:=$(shell seq 0 ${fragnum})
tfold:=test/$(method)
ifdef chunk
    seq:=x
else
    seq:=$($(set)seq)
endif
prefix:=$($(set)pre)

.PHONY: all

all: $(seq:%=$(tfold)/dist-$(set)/hist-%.pdf) $(tfold)/dist-$(set)/record.csv
$(tfold)/dist-$(set)/record.csv: $(seq:%=$(tfold)/dist-$(set)/record-%.csv)
	cat $^ > $@
$(tfold)/dist-$(set)/record-%.csv: $(tfold)/dist-$(set)/distrecord-%.h5
	python3 test/csv_dist.py $^ -o $@
$(tfold)/dist-$(set)/hist-%.pdf: $(tfold)/dist-$(set)/distrecord-%.h5
	python3 test/draw_dist.py $^ -o $@
$(tfold)/dist-$(set)/distrecord-%.h5: $(datfold)/$(set)/$(prefix)%.h5 $(tfold)/sub-$(set)/submission-%.h5
	@mkdir -p $(dir $@)
	python3 test/test_dist.py $(word 2,$^) --ref $< -o $@ > $@.log 2>&1
define split
$(tfold)/sub-$(set)/submission-$(1).h5: $(tfold)/sub-$(set)/total-$(1).h5
	python3 test/adjust.py $$^ -o $$@
$(tfold)/sub-$(set)/total-$(1).h5: $(fragseq:%=$(tfold)/unad-$(set)/unadjusted-$(1)-%.h5)
	@mkdir -p $$(dir $$@)
	python3 test/integrate.py $$^ --num ${fragnum} -o $$@
$(tfold)/unad-$(set)/unadjusted-$(1)-%.h5: $(datfold)/$(set)/$(prefix)$(1).h5 test/spe-$(set).h5
	@mkdir -p $$(dir $$@)
	python3 test/fit.py $$< --met $(method) --ref $$(word 2,$$^) --num ${fragnum} -o $$@
endef
$(foreach i,$(seq),$(eval $(call split,$(i))))

$(datfold)/$(set)/$(prefix)x.h5: $(datfold)/$(set)/$(prefix)$(chunk).h5
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
