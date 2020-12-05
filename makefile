.PHONY : all

runNos=$(shell cat TargetList.txt)
# flag=$(runNos:%=Preoutput/run%/.flag)
# 
# all : $(Targets)

define Link

RawDatas$(1)=$$(shell ls -1U $(JPDataDir)/run$(1)/*$(1)_*.root || ls -1U $(JPDataDir)/run$(1)/*$(1).root)
RawData_$(1)=$$(shell echo $$(firstword $$(RawDatas$(1))) | sed -E 's,$(1)_?[0-9]*\.root,$(1),')
Links$(1)=$$(patsubst $$(RawData_$(1))_%.root, raw/run$(1)/%.root, $$(RawDatas$(1)))
RawData0_$(1)=$$(RawData_$(1)).root
Link0_$(1)=raw/run$(1)/0.root
Bsln_$(1)=$$(wildcard )
LinkBsln_$(1)=$$(patsubst $$(RawData_$(1))_%.root, pre/run$(1)/%.root, $$(RawDatas$(1)))

all : $$(Links$(1)) $$(Link0_$(1)) $$(LinkBsln_$(1))

raw/run$(1)/%.root : $$(RawData_$(1))_%.root | PrepareDir_$(1)
	ln -snf $$< $$@

pre/run$(1)/%.root : /mnt/neutrino/02_PreAnalysis_wuyy/run$(1)/PreAnalysis_Run$$(shell echo $$$$(($(1)+0)))_File%.root
	ln -snf $$< $$@

$$(Link0_$(1)) : $$(RawData0_$(1)) | PrepareDir_$(1)
	if [ -e $$< ]; then ln -snf $$< $$@; fi

$$(RawData0_$(1)) :

PrepareDir_$(1) :
	@mkdir -p raw/run$(1)/
	@mkdir -p pre/run$(1)/

endef

$(info $(foreach i ,$(runNos), $(call Link,$(i))))
$(eval $(foreach i ,$(runNos), $(call Link,$(i))))

.DELETE_ON_ERROR:
