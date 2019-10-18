test/xdc/FourierTransform/single_pe.h5: ztraining-0.h5
	python3 test/xdc/FourierTransform/standard.py $^ -o $@

ztraining-0.h5: %:
	wget 'https://cloud.tsinghua.edu.cn/f/0499334a4239427798c1/?dl=1&first.h5' -O $@

# fadc_010217.000091.h5 fadc_010252.000180.h5 fadc_013110.000014.h5 PoC-problem.h5: %:
#	wget http://hep.tsinghua.edu.cn/~orv/distfiles/$@

.DELETE_ON_ERROR:

.SECONDARY:
