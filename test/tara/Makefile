.PHONY: all
# answer filename
#ansfile = ../../ztraining-1.h5
ansfile = ../../abtraining-9.h5
# problem filename
#problemfile = ../../ztraining-1.h5
problemfile = ../../abtraining-9.h5
# name of the directory will be made to store the result
fileStore = abtraining-9
# submission file will be generated
subfile = Prediction_Results/Prediction_Mod_ztraining-1.h5
# method name and will use in the filename prefix
method = TaraZtrain

# training filename
#trainingfile = ../../ztraining-0.h5
trainingfile = ../../abtraining-9.h5
# model name prefix
trainStore =abtraining-9

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
model: $(trainingfile)
	python3 Data_Pre-Processing.py $^ $(trainStore) 
	python3 Data_Processing.py $(trainStore) $(trainStore) 

.DELETE_ON_ERROR:

.SECONDARY:
