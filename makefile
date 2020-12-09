runNos=$(shell cat TargetList.txt)

define Analysis

raw_$(1)=$(wildcard raw/run$(1)/*.root)

analysis_$(1)=$$(patsubst raw/run$(1)/%.root,/mnt/eternity/cnn_charge/run$(1)/%.h5,$$(raw_$(1)))

Analysis : $$(analysis_$(1))

$$(analysis_$(1)) : $$(raw_$(1))

$$(firstword $$(analysis_$(1))) : $$(firstwordr $$(raw_$(1)))
	mkdir -p $$(dir $$@)
	time python3.8 -u Prediction_Processing_Total.py $$(dir $$<) /mnt/eternity/baseline/run$(1) $(shell ./get_gaintable.py $(1)) -N old1029/ --met charge --device 0 -o $$(dir $$@) -B 17000 > $$(dir $$@)/analysis.log 2>&1

endef

$(eval $(foreach r,$(runNos),$(call Analysis,$(r))))


define Link

RawDatas$(1)=$$(shell ls -1U $(JPDataDir)/run$(1)/*$(1)_*.root || ls -1U $(JPDataDir)/run$(1)/*$(1).root)
RawData_$(1)=$$(shell echo $$(firstword $$(RawDatas$(1))) | sed -E 's,$(1)_?[0-9]*\.root,$(1),')
Links$(1)=$$(patsubst $$(RawData_$(1))_%.root, raw/run$(1)/%.root, $$(RawDatas$(1)))
RawData0_$(1)=$$(RawData_$(1)).root
Link0_$(1)=raw/run$(1)/0.root
Bsln_$(1)=$$(wildcard )
LinkBsln_$(1)=$$(patsubst $$(RawData_$(1))_%.root, pre/run$(1)/%.root, $$(RawDatas$(1)))

Link : $$(Links$(1)) $$(Link0_$(1))

raw/run$(1)/%.root : $$(RawData_$(1))_%.root | PrepareDir_$(1)
	ln -snf $$< $$@

$$(Link0_$(1)) : $$(RawData0_$(1)) | PrepareDir_$(1)
	if [ -e $$< ]; then ln -snf $$< $$@; fi

$$(RawData0_$(1)) :

PrepareDir_$(1) :
	@mkdir -p raw/run$(1)/

endef

# $(info $(foreach i ,$(runNos), $(call Link,$(i))))
$(eval $(foreach i ,$(runNos), $(call Link,$(i))))

.DELETE_ON_ERROR:
