SHELL:=bash
channelN:=$(shell seq -f '%02g' 0 29)
raw:=$(wildcard $(datfold)/*.h5)
charge:=$(patsubst $(datfold)/%.h5,$(datfold)/wave_%.h5,$(raw))

tfold:=$(method)-$(mode)
ifdef chunk
    seq:=$(chunk)x
    datfoldi:=dataset
else
    datfoldi:=$(datfold)
endif
ifeq ($(method), takara)
    predict:=nn
else
    predict:=fit
endif

PreData:=$(channelN:%=$(datfold)/PreProcess/Pre_Channel%.h5)
Nets:=$(channelN:%=$(datfold)/$(mode)/Nets/Channel%.torch_net)

.PHONY : all

all : $(seq:%=$(tfold)/dist/hist-%.pdf) $(seq:%=$(tfold)/dist/record-%.csv)
$(tfold)/dist/record-%.csv : $(tfold)/dist/distr-%.h5
	python3 csv_dist.py $^ --mod $(mode) -o $@
$(tfold)/dist/hist-%.pdf : $(tfold)/dist/distr-%.h5
	python3 draw_dist.py $^ --mod $(mode) -o $@
$(tfold)/dist/distr-%.h5 : $(datfoldi)/$(prefix)%.h5 $(tfold)/resu/sub-%.h5 spe.h5
	@mkdir -p $(dir $@)
	python3 test_dist.py $(word 2,$^) --mod $(mode) --ref $< $(word 3,$^) -o $@ > $@.log 2>&1
define fit
$(tfold)/resu/sub-$(1).h5 : $(datfoldi)/$(prefix)$(1).h5 spe.h5
	@mkdir -p $$(dir $$@)
	export OMP_NUM_THREADS=2 && python3 fit.py $$< --mod $(mode) --met $(method) --ref $$(wordlist 2,3,$$^) -o $$@ > $$@.log 2>&1
endef

define nn
$(tfold)/resu/sub-$(1).h5 : $(datfoldi)/$(prefix)$(1).h5 spe.h5 $(Nets)
	@mkdir -p $$(dir $$@)
	python3 -u Prediction_Processing_Total.py $$< --mod $(mode) --met $(method) -o $$@ --ref $$(word 2,$$^) -N $(datfold)/$(mode)/Nets -D 0 > $$@.log 2>&1
endef
$(foreach i,$(seq),$(eval $(call $(predict),$(i))))

model : $(Nets)

$(datfold)/$(mode)/Nets/Channel%.torch_net : $(datfold)/$(mode)/$(prefix)Channel%/.Training_finished ;

$(datfold)/$(mode)/$(prefix)Channel%/.Training_finished : $(datfold)/PreProcess/Pre_Channel%.h5 spe.h5
	@mkdir -p $(dir $@)
	@mkdir -p $(datfold)/$(mode)/Nets
	python3 -u Data_Processing.py $< -n $* -B 64 --mod $(mode) --ref $(word $(words $^), $^) -o $(datfold)/$(mode)/Nets/Channel$*.torch_net $(dir $@) > $(dir $@)Train.log 2>&1
	@touch $@

$(PreData) : $(datfold)/.PreProcess

$(datfold)/.PreProcess : $(seq:%=$(datfold)/$(prefix)%.h5) spe.h5
	@mkdir -p $(datfold)/PreProcess
	python3 -u Data_Pre-Processing.py $(datfold)/$(prefix) -o $(datfold)/PreProcess/Pre_Channel --ref $(word $(words $^), $^) > $(datfold)/PreProcess/PreProcess.log 2>&1
	@touch $@

$(datfoldi)/$(prefix)$(chunk)x.h5 : $(datfold)/$(prefix)$(chunk).h5
	@mkdir -p $(dir $@)
	python3 cut_data.py $^ -o $@ -a -1 -b 10000

spe.h5 : $(datfold)/$(prefix)0.h5
	python3 spe_get.py $^ -o $@ --num 500000 --len 80

model.pkl :
	python3 model.py $@

.DELETE_ON_ERROR: 

.SECONDARY:
