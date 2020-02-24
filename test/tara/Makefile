.PHONY : help 

usage : help

.PHONY : usage

help:
	@cat ./mannual.txt

# training fileno
TrainFileNo = 0
# training fileno
TestFileNo = *
# max training set
MaxSet = 100000
TrainDataDir = ../../dataset/jinp/
NetStorageDir = ./
TestDataDir = $(TrainDataDir)
ResultStorageDir = ./
#training filename
traindata_name_prefix = ztraining-
trainfile_prefix = $(TrainDataDir)$(traindata_name_prefix)
trainfile = $(wildcard $(trainfile_prefix)$(TrainFileNo).h5)
#test filename
testdata_name_prefix = ztraining-
testfile_prefix = $(TestDataDir)$(testdata_name_prefix)
testfile = $(wildcard $(testfile_prefix)$(TestFileNo).h5)
# Result storage name
Result_prefix = $(ResultStorageDir)Analyzing_Results_$(testdata_name_prefix)
Result = $(patsubst $(testfile_prefix)%.h5,$(Result_prefix)%/.Analyzing_finished, $(testfile))
# Network Models storage name
NetStore_prefix = $(NetStorageDir)Network_Models_$(traindata_name_prefix)
NetStore = $(patsubst $(trainfile_prefix)%.h5,$(NetStore_prefix)%/.Training_finished, $(trainfile))


.PHONY : Grade Analysis model .Bulletin

Grade : $(patsubst $(testfile_prefix)%.h5,$(Result_prefix)%/DistanceDistribution.png,$(testfile))

Analysis : $(Result)

model : $(NetStore)

.Bulletin:
	rm -f ./.bulletin.swp

$(Result_prefix)%/.Analyzing_finished : $(word 2,$(NetStore)) $(testfile_prefix)%.h5 | .Bulletin
	mkdir -p $(dir $@)
	python3 -u Prediction_Processing_Total.py $(dir $<) $(word 1,$^) $(dir $@) $* > $(dir $@)Analysis.log 2>&1
	touch $@

$(Result_prefix)%/DistanceDistribution.png: $(Result_prefix)%/DistanceDistribution.npy
	python3 plotDistribution.py $^ $@

$(Result_prefix)%/DistanceDistribution.npy : $(Result_prefix)%/.Analyzing_finished $(testfile_prefix)%.h5
	python3 graderMain.py $(Result_prefix)$*/ $(dir $<)Prediction.h5 $(word 2,$^)

$(NetStore_prefix)%/.PreProcessing_finished : $(trainfile_prefix)%.h5
	mkdir -p $(dir $@)
	python3 -u Data_Pre-Processing.py $< $(dir $@) $(MaxSet) > $(dir $@)PreProcess.log 2>&1
	touch $@

$(NetStore_prefix)%/.Training_finished : $(NetStore_prefix)%/.PreProcessing_finished | .Bulletin
	python3 -u Data_Processing.py $(dir $@) $(dir $<) $(MaxSet) $* > $(dir $@)Train.log  2>&1
	touch $@

.DELETE_ON_ERROR: 

.SECONDARY:
