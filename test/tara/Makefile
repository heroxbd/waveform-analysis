.PHONY : help 

.PHONY : usage

help:
	@cat ./mannual.txt
# answer filename
# ansfile = ../../abanswer.h5
# problem filename
# problemfile = ../../abtest.h5
# name of the directory will be made to store the result
# fileStore = abtraining-9
# submission file will be generated
# subfile = Prediction_Results/Prediction_Mod_abanswer.h5
# method name and will use in the filename prefix
method = TaraZtrain

# training fileno
TrainFileNo = *
# training fileno
TestFileNo = *
# max training set
MaxSet = 100000
TrainDataDir = dataset/jinp/
NetWorkDir = $(TrainDataDir)
TestDataDir = dataset/jinp/
ResultDir = $(TestDataDir)
#training filename
traindata_name_prefix = ztraining-
trainfile_prefix = $(TrainDataDir)$(traindata_name_prefix)
trainfile_suffix = .h5
trainfile = $(wildcard $(trainfile_prefix)$(TrainFileNo)$(trainfile_suffix))
#test filename
testdata_name_prefix = zincm-problem-
testfile_prefix = $(TestDataDir)$(testdata_name_prefix)
testfile_suffix = .h5
testfile = $(wildcard $(testfile_prefix)$(TestFileNo)$(testfile_suffix))
# PreAnalysis storage name
PreAnalysis_prefix = $(ResultDir)Pre-Processing_Results_$(testdata_name_prefix)
PreAnalysis_suffix = / 
PreAnalysis_flag = /.finshed
PreAnalysis = $(patsubst $(testfile_prefix)%$(testfile_suffix),$(PreAnalysis_prefix)%$(PreAnalysis_flag), $(testfile))
# Result storage name
Result_prefix = $(ResultDir)Analyzing_Results_$(testdata_name_prefix)
Result_suffix = / 
Result_flag = /.finshed
Result = $(patsubst $(testfile_prefix)%$(testfile_suffix),$(Result_prefix)%$(Result_flag), $(testfile))
# PreProcessing storage name
PreStore_prefix = $(NetWorkDir)Pre-Processing_Results_$(traindata_name_prefix)
PreStore_suffix = / 
PreStore_flag = /.finshed
PreStore = $(patsubst $(trainfile_prefix)%$(trainfile_suffix),$(PreStore_prefix)%$(PreStore_flag), $(trainfile))
# Network Models storage name
NetStore_prefix = $(NetWorkDir)Network_Models_$(traindata_name_prefix)
NetStore_suffix = / 
NetStore_flag = /.finshed
NetStore = $(patsubst $(trainfile_prefix)%$(trainfile_suffix),$(NetStore_prefix)%$(NetStore_flag), $(trainfile))
# test log name
testlog_prefix = $(testdata_name_prefix)
testlog_suffix = .log 
testlog = $(patsubst $(testfile_prefix)%$(testfile_suffix),$(testlog_prefix)%$(testlog_suffix), $(testfile))
# training log name
trainlog_prefix = $(traindata_name_prefix)
trainlog_suffix = .log 
trainlog = $(patsubst $(trainfile_prefix)%$(trainfile_suffix),$(trainlog_prefix)%$(trainlog_suffix), $(trainfile))
# finish flag
# finish_pro = $(patsubst $(trainfile_prefix)%$(trainfile_suffix),.finish_pro%.swp, $(trainfile))

Analysis : $(Result)

$(Result_prefix)%$(Result_flag) : $(word 1,$(NetStore)) $(PreAnalysis_prefix)%$(PreAnalysis_flag)
	python3 -u Prediction_Processing_Total.py $(dir $<) $(dir $(word 2,$^)) $(dir $@) > $(testlog_prefix)_Train-$*$(testlog_suffix) 2>&1

$(PreAnalysis_prefix)%$(PreAnalysis_flag) : $(testfile_prefix)%$(testfile_suffix)
	python3 -u Prediction_Pre-Processing.py $< $(dir $@) > $(testlog_prefix)_Pre-$*$(testlog_suffix) 2>&1
	touch $@

#$(Result_prefix)%$(Result_suffix)DistanceDistribution.png: $(Result_prefix)%$(Result_suffix)DistanceDistribution.npy
#	python3 plotDistribution.py $^ $@

$(Result_prefix)%$(Result_suffix)DistanceDistribution.npy: $(subfile) $(ansfile)
	python3 graderMain.py $(fileStore)/$(method) $^ $@

.PHONY: model
model:  $(NetStore)

$(PreStore_prefix)%$(PreStore_flag) : $(trainfile_prefix)%$(trainfile_suffix)
	python3 -u Data_Pre-Processing.py $< $(dir $@) $(MaxSet) > $(trainlog_prefix)_Pre-$*$(trainlog_suffix) 2>&1
	touch $@

$(NetStore_prefix)%$(NetStore_flag) : $(PreStore_prefix)%$(PreStore_flag) .Bulletin 
	python3 -u Data_Processing.py $(dir $@) $(dir $<) $(MaxSet) $* > $(trainlog_prefix)_Train-$*$(trainlog_suffix) 2>&1
	touch $@

.Bulletin : 
	rm -f ./.bulletin.swp

.DELETE_ON_ERROR: 

.SECONDARY:

# .finish%.swp :
# 	@touch $@

#refresh_PreStore : 
#	@touch $(patsubst $(trainlog_prefix)%$(trainlog_suffix),.finish_pre%.swp, $(trainlog))
#	@touch $(PreStore)
#
#refresh_NetStore : 
#	@touch $(patsubst $(trainlog_prefix)%$(trainlog_suffix),.finish_pro%.swp, $(trainlog))
#	@touch $(NetStore)
