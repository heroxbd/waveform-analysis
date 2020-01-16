SHELL=bash
jinpDir=dataset/jinp
jinpwaveseq=$(shell seq 0 9)
junoDir=dataset/juno
junowaveseq=2 4
xdcFTp=test/xdcFT
xiaoPp=test/xiaopeip
xiaoPl=99
xiaoPseq=$(shell seq 0 ${xiaoPl})
lucySrcDir=test/lucyddm
lucyOutDir=output/jinping/lucyddm

.PHONY:xdcFT xiaopeip lucyddm jinpDataset junoDataset

xiaopeip: $(jinpwaveseq:%=$(xiaoPp)/hist-%.pdf) $(xiaoPp)/record.csv
$(xiaoPp)/record.csv: $(jinpwaveseq:%=$(xiaoPp)/record/record-%.csv)
	cat $^ > $@
$(xiaoPp)/record/record-%.csv: $(xiaoPp)/distrecord/distrecord-%.h5
	mkdir -p $(dir $@)
	python3 test/csv_dist.py $^ -o $@
$(xiaoPp)/hist-%.pdf: $(xiaoPp)/distrecord/distrecord-%.h5
	python3 test/draw_dist.py $^ --wthres 10 -o $@
$(xiaoPp)/distrecord/distrecord-%.h5: $(jinpDir)/ztraining-%.h5 $(xiaoPp)/submission/submission-%.h5
	mkdir -p $(dir $@)
	python3 test/test_dist.py $(word 2,$^) --ref $< -o $@
define xpp_split
$(xiaoPp)/submission/submission-$(1).h5: $(xiaoPp)/submission/adjusted-$(1).h5
	mkdir -p $$(dir $$@)
	python3 $(xiaoPp)/adjust.py $$^ -o $$@
$(xiaoPp)/submission/adjusted-$(1).h5: $(xiaoPseq:%=$(xiaoPp)/unadjusted/unadjusted-$(1)-%.h5)
	mkdir -p $$(dir $$@)
	python3 $(xiaoPp)/integrate.py $$^ --num ${xiaoPl} -o $$@
$(xiaoPp)/unadjusted/unadjusted-$(1)-%.h5: $(jinpDir)/ztraining-$(1).h5 $(xiaoPp)/averspe.h5
	mkdir -p $$(dir $$@)
	python3 $(xiaoPp)/finalfit.py $$< --ref $$(word 2,$$^) --num ${xiaoPl} -o $$@
endef
$(foreach i,$(jinpwaveseq),$(eval $(call xpp_split,$(i))))
$(xiaoPp)/averspe.h5: $(jinpDir)/ztraining-0.h5
	python3 test/spe_get.py $^ -o $@

xdcFT: $(jinpwaveseq:%=$(xdcFTp)/hist-%.pdf) $(xdcFTp)/record.csv
$(xdcFTp)/record.csv: $(jinpwaveseq:%=$(xdcFTp)/record/record-%.csv)
	cat $^ > $@
$(xdcFTp)/record/record-%.csv: $(xdcFTp)/distrecord/distrecord-%.h5
	mkdir -p $(dir $@)
	python3 test/csv_dist.py $^ -o $@
$(xdcFTp)/hist-%.pdf: $(xdcFTp)/distrecord/distrecord-%.h5
	python3 test/draw_dist.py $^ --wthres 10 -o $@
$(xdcFTp)/distrecord/distrecord-%.h5: $(jinpDir)/ztraining-%.h5 $(xdcFTp)/submission/submission-%.h5
	mkdir -p $(dir $@)
	python3 test/test_dist.py $(word 2,$^) --ref $< -o $@
$(xdcFTp)/submission/submission-%.h5 : $(jinpDir)/ztraining-%.h5 $(xdcFTp)/single_pe.h5
	mkdir -p $(dir $@)
	python3 $(xdcFTp)/FFT_decon.py $< --ref $(word 2,$^) -o $@ -k 0.05 -a 4 -e 4
$(xdcFTp)/single_pe.h5: $(jinpDir)/ztraining-0.h5
	python3 test/spe_get.py $^ -o $@

lucyddm: $(jinpwaveseq:%=$(lucyOutDir)/hist-%.pdf) $(lucyOutDir)/record.csv
$(lucyOutDir)/record.csv: $(jinpwaveseq:%=$(lucyOutDir)/record/record-%.csv)
	cat $^ > $@
$(lucyOutDir)/record/record-%.csv: $(lucyOutDir)/distrecord/distrecord-%.h5
	mkdir -p $(dir $@)
	python3 test/csv_dist.py $^ -o $@
$(lucyOutDir)/hist-%.pdf: $(lucyOutDir)/distrecord/distrecord-%.h5
	python3 test/draw_dist.py $^ --wthres 10 -o $@
$(lucyOutDir)/distrecord/distrecord-%.h5: $(jinpDir)/ztraining-%.h5 $(lucyOutDir)/submission/submission-%.h5
	mkdir -p $(dir $@)
	python3 test/test_dist.py $(word 2,$^) --ref $< -o $@
$(lucyOutDir)/submission/submission-%.h5 : $(jinpDir)/ztraining-%.h5 $(lucyOutDir)/spe.h5
	mkdir -p $(dir $@)
	python3 $(lucySrcDir)/lucyDDM.py $^ $@ > $@.log 2>&1
$(lucyOutDir)/spe.h5: $(jinpDir)/ztraining-0.h5
	mkdir -p $(dir $@)
	python3 $(lucySrcDir)/speGet.py $^ $@

jinpDataset: $(jinpwaveseq:%=$(jinpDir)/ztraining-%.h5) $(jinpDir)/zincm-problem.h5
$(jinpDir)/zincm-problem.h5:
	mkdir -p $(dir $@)
	wget 'https://cloud.tsinghua.edu.cn/f/3babd73926ce47c8893a/?dl=1&first.h5' -O $@
$(jinpDir)/ztraining-9.h5:
	mkdir -p $(dir $@)
	wget 'https://cloud.tsinghua.edu.cn/f/04dca9b735494acd9781/?dl=1&first.h5' -O $@
$(jinpDir)/ztraining-8.h5:
	mkdir -p $(dir $@)
	wget 'https://cloud.tsinghua.edu.cn/f/7f5b09e096e0466c804f/?dl=1&first.h5' -O $@
$(jinpDir)/ztraining-7.h5:
	mkdir -p $(dir $@)
	wget 'https://cloud.tsinghua.edu.cn/f/cf4359cb07a846bc94f8/?dl=1&first.h5' -O $@
$(jinpDir)/ztraining-6.h5:
	mkdir -p $(dir $@)
	wget 'https://cloud.tsinghua.edu.cn/f/d4d2cd83b7084f7e8672/?dl=1&first.h5' -O $@
$(jinpDir)/ztraining-5.h5:
	mkdir -p $(dir $@)
	wget 'https://cloud.tsinghua.edu.cn/f/acc278feed464cafad14/?dl=1&first.h5' -O $@
$(jinpDir)/ztraining-4.h5:
	mkdir -p $(dir $@)
	wget 'https://cloud.tsinghua.edu.cn/f/00b99bfb7a2e411f8c54/?dl=1&first.h5' -O $@
$(jinpDir)/ztraining-3.h5:
	mkdir -p $(dir $@)
	wget 'https://cloud.tsinghua.edu.cn/f/9926a7c35a934974872f/?dl=1&first.h5' -O $@
$(jinpDir)/ztraining-2.h5:
	mkdir -p $(dir $@)
	wget 'https://cloud.tsinghua.edu.cn/f/a445fe1b4bf74357b26d/?dl=1&first.h5' -O $@
$(jinpDir)/ztraining-1.h5:
	mkdir -p $(dir $@)
	wget 'https://cloud.tsinghua.edu.cn/f/ca29b48212e44332a215/?dl=1&first.h5' -O $@
$(jinpDir)/ztraining-0.h5:
	mkdir -p $(dir $@)
	wget 'https://cloud.tsinghua.edu.cn/f/0499334a4239427798c1/?dl=1&first.h5' -O $@

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
