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
ResultDir = $(TrainDataDir)
TestDataDir = $(TrainDataDir)
#training filename
train_data_name_prefix = abtraining-
trainfile_prefix = $(TrainDataDir)$(train_data_name_prefix)
trainfile_suffix = .h5
trainfile = $(wildcard $(trainfile_prefix)$(TrainFileNo)$(trainfile_suffix))
#test filename
test_data_name_prefix = zincm-problem-
testfile_prefix = $(TestDataDir)$(test_data_name_prefix)
testfile_suffix = .h5
testfile = $(wildcard $(testfile_prefix)$(TestFileNo)$(testfile_suffix))
# PreAnalysis storage name
PreAnalysis_prefix = $(NetWorkDir)Pre-Processing_Results_$(test_data_name_prefix)
PreAnalysis_suffix = / 
PreAnalysis = $(patsubst $(testfile_prefix)%$(testfile_suffix),$(PreAnalysis_prefix)%$(PreAnalysis_flag), $(testfile))
PreAnalysis_flag = /.finshed
# PreProcessing storage name
PreStore_prefix = $(NetWorkDir)Pre-Processing_Results_$(train_data_name_prefix)
PreStore_suffix = / 
PreStore = $(patsubst $(trainfile_prefix)%$(trainfile_suffix),$(PreStore_prefix)%$(PreStore_flag), $(trainfile))
PreStore_flag = /.finshed
# Network Models storage name
NetStore_prefix = $(NetWorkDir)Network_Models_$(train_data_name_prefix)
NetStore_suffix = / 
NetStore_flag = /.finshed
NetStore = $(patsubst $(trainfile_prefix)%$(trainfile_suffix),$(NetStore_prefix)%$(NetStore_flag), $(trainfile))
# test log name
testlog_prefix = $(test_data_name_prefix)
testlog_suffix = .log 
testlog = $(patsubst $(testfile_prefix)%$(testfile_suffix),$(testlog_prefix)%$(testlog_suffix), $(testfile))
# training log name
trainlog_prefix = $(train_data_name_prefix)
trainlog_suffix = .log 
trainlog = $(patsubst $(trainfile_prefix)%$(trainfile_suffix),$(trainlog_prefix)%$(trainlog_suffix), $(trainfile))
# finish flag
# finish_pro = $(patsubst $(trainfile_prefix)%$(trainfile_suffix),.finish_pro%.swp, $(trainfile))


#Result: $(fileStore)/$(method)DistanceDistribution.png
PreProcess: $(PreAnalysis)

$(PreAnalysis_prefix)%$(PreAnalysis_flag) : $(testfile_prefix)%$(testfile_suffix)
	python3 -u Prediction_Pre-Processing.py $< $(PreAnalysis_prefix)$*$(PreAnalysis_suffix) > $(testlog_prefix)_Pre-$*$(testlog_suffix) 2>&1
	touch $@

# $(fileStore)/$(method)DistanceDistribution.png: $(fileStore)/$(method)DistanceDistribution.npy
# 	python3 plotDistribution.py $^ $@

# $(fileStore)/$(method)DistanceDistribution.npy: $(subfile) $(ansfile)
# 	python3 graderMain.py $(fileStore)/$(method) $^ $@
# $(subfile): $(problemfile)
# 	mkdir -p $(fileStore)
# 	python3 Prediction_Pre-Processing.py $(problemfile) $(fileStore)
# 	python3 Prediction_Processing_Total.py $(problemfile) $(fileStore)
	# python3 Continue_Training.py $(problemfile) $(fileStore)

.PHONY: model
model:  $(NetStore)

$(PreStore_prefix)%$(PreStore_flag) : $(trainfile_prefix)%$(trainfile_suffix)
	python3 -u Data_Pre-Processing.py $< $(PreStore_prefix)$*$(PreStore_suffix) $(MaxSet) > $(trainlog_prefix)_Pre-$*$(trainlog_suffix) 2>&1
	touch $@

$(NetStore_prefix)%$(NetStore_flag) : $(PreStore_prefix)%$(PreStore_flag) .Bulletin 
	python3 -u Data_Processing.py $(NetStore_prefix)$*$(NetStore_suffix) $(PreStore_prefix)$*$(PreStore_suffix) $(MaxSet) $* > $(trainlog_prefix)_Train-$*$(trainlog_suffix) 2>&1
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
