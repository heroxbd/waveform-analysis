cids=$(shell seq -f %02g 0 29)

targets=$(cids:%=net%/.trained)

net%/.trained : /data/wuyy/GH1/PreProcess/%.h5
	mkdir -p $(dir $@)
	python3 -u Data_Processing.py /data/wuyy/GH1/PreProcess/$*.h5 -o charge $(dir $@) --mod Charge -n $* > $(dir $@)train.log 2>&1 
	touch $@


all : $(targets)
