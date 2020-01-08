SHELL=bash
range0:=$(shell echo {0..9})
xdcFTp=test/xdc/FT
xiaoPp=test/xiaopeip

.PHONY:all0 all1 lucyddm

all: all1

all1: $(range0:%=$(xiaoPp)/hist-%.pdf) $(xiaoPp)/record.csv

$(xiaoPp)/record.csv: $(range0:%=$(xiaoPp)/record/record-%.csv)
	cat $^ > $@

$(xiaoPp)/record/record-%.csv: $(xiaoPp)/distrecord/distrecord-%.h5
	mkdir -p $(dir $@)
	python3 test/csv_dist.py $^ -o $@

$(xiaoPp)/hist-%.pdf: $(xiaoPp)/distrecord/distrecord-%.h5
	python3 test/draw_dist.py $^ -o $@

$(xiaoPp)/distrecord/distrecord-%.h5: ztraining-%.h5 $(xiaoPp)/submission/submission-%.h5
	mkdir -p $(dir $@)
	python3 test/test_dist.py $(word 2,$^) --ref $< -o $@

$(xiaoPp)/submission/submission-%.h5: $(xiaoPp)/unadjusted/unadjusted-%.h5
	mkdir -p $(dir $@)
	python3 $(xiaoPp)/adjust.py $^ -o $@

$(xiaoPp)/unadjusted/unadjusted-%.h5: ztraining-%.h5 $(xiaoPp)/averspe.h5
	mkdir -p $(dir $@)
	python3 $(xiaoPp)/finalfit.py $< --ref $(word 2,$^) -o $@

$(xiaoPp)/averspe.h5: ztraining-0.h5
	python3 $(xiaoPp)/read.py $^ -o $@

all0: $(range0:%=$(xdcFTp)/hist-%.pdf) $(xdcFTp)/record.csv

$(xdcFTp)/record.csv: $(range0:%=$(xdcFTp)/record/record-%.csv)
	cat $^ > $@

$(xdcFTp)/record/record-%.csv: $(xdcFTp)/distrecord/distrecord-%.h5
	mkdir -p $(dir $@)
	python3 test/csv_dist.py $^ -o $@

$(xdcFTp)/hist-%.pdf: $(xdcFTp)/distrecord/distrecord-%.h5
	python3 test/draw_dist.py $^ -o $@

$(xdcFTp)/distrecord/distrecord-%.h5: ztraining-%.h5 $(xdcFTp)/submission/submission-%.h5
	mkdir -p $(dir $@)
	python3 test/test_dist.py $(word 2,$^) --ref $< -o $@

$(xdcFTp)/submission/submission-%.h5 : ztraining-%.h5 $(xdcFTp)/single_pe.h5
	mkdir -p $(dir $@)
	python3 $(xdcFTp)/FFT_decon.py $< --ref $(word 2,$^) -o $@

$(xdcFTp)/single_pe.h5: $(range0:%=ztraining-%.h5)
	python3 $(xdcFTp)/standard.py $^ -o $@

lucyOutDir=output/jinping/lucyddm
lucySrcDir=test/lucyddm
lucyddm: $(range0:%=$(lucyOutDir)/hist-%.pdf) $(lucyOutDir)/record.csv

$(lucyOutDir)/record.csv: $(range0:%=$(lucyOutDir)/record/record-%.csv)
	cat $^ > $@

$(lucyOutDir)/record/record-%.csv: $(lucyOutDir)/distrecord/distrecord-%.h5
	mkdir -p $(dir $@)
	python3 test/csv_dist.py $^ -o $@

$(lucyOutDir)/hist-%.pdf: $(lucyOutDir)/distrecord/distrecord-%.h5
	python3 test/draw_dist.py $^ -o $@

$(lucyOutDir)/distrecord/distrecord-%.h5: ztraining-%.h5 $(lucyOutDir)/submission/submission-%.h5
	mkdir -p $(dir $@)
	python3 test/test_dist.py $(word 2,$^) --ref $< -o $@

$(lucyOutDir)/submission/submission-%.h5 : ztraining-%.h5 $(lucyOutDir)/spe.h5
	mkdir -p $(dir $@)
	python3 $(lucySrcDir)/lucyDDM.py $^ $@ > $@.log 2>&1

$(lucyOutDir)/spe.h5: ztraining-0.h5
	mkdir -p $(dir $@)
	python3 $(lucySrcDir)/speGet.py $^ $@ >$@.log 2>&1
zincm-problem.h5:
	wget 'https://cloud.tsinghua.edu.cn/f/3babd73926ce47c8893a/?dl=1&first.h5' -O $@

ztraining-9.h5:
	wget 'https://cloud.tsinghua.edu.cn/f/04dca9b735494acd9781/?dl=1&first.h5' -O $@

ztraining-8.h5:
	wget 'https://cloud.tsinghua.edu.cn/f/7f5b09e096e0466c804f/?dl=1&first.h5' -O $@

ztraining-7.h5:
	wget 'https://cloud.tsinghua.edu.cn/f/cf4359cb07a846bc94f8/?dl=1&first.h5' -O $@

ztraining-6.h5:
	wget 'https://cloud.tsinghua.edu.cn/f/d4d2cd83b7084f7e8672/?dl=1&first.h5' -O $@

ztraining-5.h5:
	wget 'https://cloud.tsinghua.edu.cn/f/acc278feed464cafad14/?dl=1&first.h5' -O $@

ztraining-4.h5:
	wget 'https://cloud.tsinghua.edu.cn/f/00b99bfb7a2e411f8c54/?dl=1&first.h5' -O $@

ztraining-3.h5:
	wget 'https://cloud.tsinghua.edu.cn/f/9926a7c35a934974872f/?dl=1&first.h5' -O $@

ztraining-2.h5:
	wget 'https://cloud.tsinghua.edu.cn/f/a445fe1b4bf74357b26d/?dl=1&first.h5' -O $@

ztraining-1.h5:
	wget 'https://cloud.tsinghua.edu.cn/f/ca29b48212e44332a215/?dl=1&first.h5' -O $@

ztraining-0.h5:
	wget 'https://cloud.tsinghua.edu.cn/f/0499334a4239427798c1/?dl=1&first.h5' -O $@

JUNO-Kaon-50.h5:
	wget http://hep.tsinghua.edu.cn/~orv/distfiles/JUNO-Kaon-50.h5

.PHONY: junoDataset

junoDir=dataset/juno
junowaveseq=1 3
junoDataset: $(junowaveseq:%=$(junoDir)/junoWave%.h5)
$(junoDir)/junoWave1.h5:
	wget https://cloud.tsinghua.edu.cn/f/496e083a78a94251b623/?dl=1 -O $@
$(junoDir)/junoWave3.h5:
	wget https://cloud.tsinghua.edu.cn/f/56e8ca3d3d30414da095/?dl=1 -O $@
.DELETE_ON_ERROR:

.SECONDARY:
