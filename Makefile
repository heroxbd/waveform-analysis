SHELL:=bash
jinpseq:=$(shell seq 0 0)
junoseq:=2 4
seq:=$($(set)seq)
dir:=dataset
fragnum:=0
fragseq:=$(shell seq 0 ${fragnum})
folder:=test/$(method)

.PHONY: all

all: $(seq:%=$(folder)/dist-$(set)/hist-%.pdf) $(folder)/dist-$(set)/record.csv
$(folder)/dist-$(set)/record.csv: $(seq:%=$(folder)/dist-$(set)/record-%.csv)
	cat $^ > $@
$(folder)/dist-$(set)/record-%.csv: $(folder)/dist-$(set)/distrecord-%.h5
	python3 test/csv_dist.py $^ -o $@
$(folder)/dist-$(set)/hist-%.pdf: $(folder)/dist-$(set)/distrecord-%.h5
	python3 test/draw_dist.py $^ -o $@
$(folder)/dist-$(set)/distrecord-%.h5: $(dir)/$(set)/*%.h5 $(folder)/sub-$(set)/submission-%.h5
	mkdir -p $(dir $@)
	python3 test/test_dist.py $(word 2,$^) --ref $< -o $@ > $@.log 2>&1
define split
$(folder)/sub-$(set)/submission-$(1).h5: $(folder)/sub-$(set)/total-$(1).h5
	python3 test/adjust.py $$^ -o $$@
$(folder)/sub-$(set)/total-$(1).h5: $(fragseq:%=$(folder)/unad-$(set)/unadjusted-$(1)-%.h5)
	mkdir -p $$(dir $$@)
	python3 test/integrate.py $$^ --num ${fragnum} -o $$@
$(folder)/unad-$(set)/unadjusted-$(1)-%.h5: $(dir)/$(set)/*$(1).h5 test/spe-$(set).h5
	mkdir -p $$(dir $$@)
	python3 test/fit.py $$< --met $(method) --ref $$(word 2,$$^) --num ${fragnum} -o $$@
endef
$(foreach i,$(seq),$(eval $(call split,$(i))))

test/spe-$(set).h5: $(dir)/$(set)/*$(word 1,$(seq)).h5
	python3 test/spe_get.py $^ -o $@ --num 10 --len 80

JUNO-Kaon-50.h5:
	wget http://hep.tsinghua.edu.cn/~orv/distfiles/JUNO-Kaon-50.h5

junoDataset: $(seq:%=$(junodir)/junoWave%.h5)
$(junodir)/junoWave2.h5:
	mkdir -p $(dir $@)
	wget https://cloud.tsinghua.edu.cn/f/f6e4cf503be542d3892f/?dl=1 -O $@
$(junodir)/junoWave4.h5:
	mkdir -p $(dir $@)
	wget https://cloud.tsinghua.edu.cn/f/846ecb6335564714902b/?dl=1 -O $@

.DELETE_ON_ERROR:
.SECONDARY:
