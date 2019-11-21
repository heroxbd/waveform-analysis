SHELL=bash
range0:=$(shell echo {0..9})
range1:=$(shell echo {0..1797})
xdcFTp=test/xdc/FT
xiaoPp=test/xiaopeip

.PHONY:all0 all1

all1: $(xiaoPp)/submission/submission.h5

$(xiaoPp)/submission/submission.h5: $(xiaoPp)/submission/unadjusted.h5
	python3 $(xiaoPp)/adjust.py $^ -o $@

$(xiaoPp)/submission/unadjusted.h5: $(range1:%=$(xiaoPp)/result/%-pgan.h5)
	mkdir -p $(dir $@)
	python3 $(xiaoPp)/alltran.py $^ -o $@

$(xiaoPp)/result/%-pgan.h5: zincm-problem.h5 $(xiaoPp)/averape.h5
	mkdir -p $(dir $@)
	wolframscript -file $(xiaoPp)/finalfit.wl $(patsubst $(xiaoPp)/result/%-pgan.h5, %, $@)

$(xiaoPp)/averape.h5: ztraining-0.h5
	python3 $(xiaoPp)/read.py $^ -o $@

all0: $(xdcFTp)/distpic.zip $(xdcFTp)/record.csv

$(xdcFTp)/record.csv: $(range0:%=$(xdcFTp)/record/record-%.csv)
	cat $^ > $@

$(xdcFTp)/record/record-%.csv: $(xdcFTp)/distrecord/distrecord-%.h5
	mkdir -p $(dir $@)
	python3 test/record_dist.py $^ -o $@

$(xdcFTp)/distpic.zip: $(range0:%=$(xdcFTp)/distpic/distpic-%.png)
	zip -j $@ $^

$(xdcFTp)/distpic/distpic-%.png: $(xdcFTp)/distrecord/distrecord-%.h5
	mkdir -p $(dir $@)
	python3 test/draw_dist.py $^ -o $@

$(xdcFTp)/distrecord/distrecord-%.h5: ztraining-%.h5 $(xdcFTp)/submission/submission-%.h5
	mkdir -p $(dir $@)
	python3 test/test_dist.py $(word 2,$^) --ref $< -o $@

$(xdcFTp)/submission/submission-%.h5 : ztraining-%.h5 $(xdcFTp)/single_pe.h5
	mkdir -p $(dir $@)
	python3 $(xdcFTp)/FFT_decon.py $< --ref $(word 2,$^) -o $@

$(xdcFTp)/single_pe.h5: $(range0:%=ztraining-%.h5)
	python3 $(xdcFTp)/standard.py $^ -o $@

zincm-problem.h5: %:
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


.DELETE_ON_ERROR:

.SECONDARY:
