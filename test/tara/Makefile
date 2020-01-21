.PHONY: all
# answer filename
ansfile = ../../abanswer.h5
# problem filename
problemfile = ../../abtest.h5
# name of the directory will be made to store the result
fileStore = abtraining-9
# submission file will be generated
subfile = Prediction_Results/Prediction_Mod_abanswer.h5
# method name and will use in the filename prefix
method = TaraZtrain

# training filename
#trainingfile = ../../ztraining-0.h5
trainingfile_prefix = ../../dataset/jinp/abtraining-
trainingfile_suffix = .h5
trainingfile = $(wildcard $(trainingfile_prefix)[$(shell seq -s "," 0 7)]$(trainingfile_suffix))
# training storage name
trainStore_prefix = abtraining-
trainStore_suffix =  
trainStore = $(patsubst $(trainingfile_prefix)%$(trainingfile_suffix),$(trainStore_prefix)%$(trainStore_suffix), $(trainingfile))
# training log name
trainlog_prefix = $(trainStore_prefix)
trainlog_suffix = .log 
trainlog = $(patsubst $(trainingfile_prefix)%$(trainingfile_suffix),$(trainlog_prefix)%$(trainlog_suffix), $(trainingfile))

all: $(fileStore)/$(method)DistanceDistribution.png

$(fileStore)/$(method)DistanceDistribution.png: $(fileStore)/$(method)DistanceDistribution.npy
	python3 plotDistribution.py $^ $@

$(fileStore)/$(method)DistanceDistribution.npy: $(subfile) $(ansfile)
	python3 graderMain.py $(fileStore)/$(method) $^ $@
$(subfile): $(problemfile)
	mkdir -p $(fileStore)
	python3 Prediction_Pre-Processing.py $(problemfile) $(fileStore)
	python3 Prediction_Processing_Total.py $(problemfile) $(fileStore)
	# python3 Continue_Training.py $(problemfile) $(fileStore)

.PHONY: model
model: $(trainlog) $(trainStore)

$(trainlog) : $(trainingfile)

$(trainStore_prefix)%$(trainStore_suffix) : $(trainingfile_prefix)%$(trainingfile_suffix) $(trainlog_prefix)%$(trainlog_suffix)
	python3 -u Data_Pre-Processing.py $< $@ > $(word 2,$^) 2>&1
	python3 -u Data_Processing.py $@ >> $(word 2,$^) 2>&1

.DELETE_ON_ERROR:

.SECONDARY:
