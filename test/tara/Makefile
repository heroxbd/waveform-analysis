.PHONY: help 

help:
	@echo 'Usage:'
	@echo '  make [FileNo=fileno] model'
	@echo '  make [FileNo=fileno] Analysis'
	@echo '       fileno: for example, [0] or [0,1,3] or [2,3,4,5] or simply [`seq -s , 2 5`]'
	@echo '  example: make FileNo=[`seq -s , 0 7`] model'
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

# training fileno
FileNo = *
#trainingfile = ../../ztraining-0.h5
trainingfile_prefix = ../../dataset/jinp/abtraining-
trainingfile_suffix = .h5
trainingfile = $(wildcard $(trainingfile_prefix)$(FileNo)$(trainingfile_suffix))
# PreProcessing storage name
PreStore_prefix = Pre-Processing_Results_abtraining-
PreStore_suffix = / 
PreStore = $(patsubst $(trainingfile_prefix)%$(trainingfile_suffix),$(PreStore_prefix)%$(PreStore_suffix), $(trainingfile))
# Network Models storage name
NetStore_prefix = Network_Models_abtraining-
NetStore_suffix = / 
NetStore = $(patsubst $(trainingfile_prefix)%$(trainingfile_suffix),$(NetStore_prefix)%$(NetStore_suffix), $(trainingfile))
# training log name
trainlog_prefix = abtraining-
trainlog_suffix = .log 
trainlog = $(patsubst $(trainingfile_prefix)%$(trainingfile_suffix),$(trainlog_prefix)%$(trainlog_suffix), $(trainingfile))
# reload training fileno and bulletin
FileNumbers = $(patsubst $(trainingfile_prefix)%$(trainingfile_suffix),%, $(trainingfile))


# all: $(fileStore)/$(method)DistanceDistribution.png

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
model: $(NetStore)

	

$(PreStore_prefix)%$(PreStore_suffix) : $(trainingfile_prefix)%$(trainingfile_suffix) .finish_pre%.swp
	python3 -u Data_Pre-Processing.py $^ $@ > $(trainlog_prefix)$*$(trainlog_suffix) 2>&1
	@touch .finish_pre$*.swp
	@touch $@

$(trainlog) : $(PreStore)

$(NetStore_prefix)%$(NetStore_suffix) : $(PreStore_prefix)%$(PreStore_suffix) $(trainlog_prefix)%$(trainlog_suffix) .finish_pro%.swp FileNumbers
	python3 -u Data_Processing.py $@ $< $* >> $(word 2,$^) 2>&1
	@touch .finish_pro$*.swp
	@touch $@

FileNumbers :
	echo $(FileNumbers) > ./.bulletin.swp

.finish%.swp :
	@touch $@

.DELETE_ON_ERROR:

.SECONDARY:

refresh_PreStore : 
	@touch $(patsubst $(trainlog_prefix)%$(trainlog_suffix),.finish_pre%.swp, $(trainlog))
	@touch $(PreStore)

refresh_NetStore : 
	@touch $(patsubst $(trainlog_prefix)%$(trainlog_suffix),.finish_pro%.swp, $(trainlog))
	@touch $(NetStore)
