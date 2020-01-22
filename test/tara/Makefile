.PHONY: help 

help:
	@echo 'Usage:'
	@echo '  make [FileNo=fileno] [DataDir=/path/to/data/dir/] [data_name_prefix=prefix] model'
	# @echo '  make [FileNo=fileno] Analysis'
	@echo '       fileno: "[%d,%d,...]" for example, [0] or [0,1,3] or [2,3,4,5] or simply [`seq -s , 2 5`]'
	@echo '       DataDir: path to data directory. The "/" at the end is essential'
	@echo '       data file name shoud obey "{prefix}%d.h5", for example abtraining-0.h5, "abtraining-" is the prefix'
	@echo '  example: make FileNo=[`seq -s , 0 7`] DataDir=/data/JUNO_DATA/Jinping_DATA_Train/dataset/jinp/ data_name_prefix=abtraining- model'
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
FileNo = *
#training filename
DataDir = /data/JUNO_DATA/Jinping_DATA_Train/dataset/jinp/
data_name_prefix = abtraining-
trainfile_prefix = $(DataDir)$(data_name_prefix)
trainfile_suffix = .h5
trainfile = $(wildcard $(trainfile_prefix)$(FileNo)$(trainfile_suffix))
# PreProcessing storage name
PreStore_prefix = $(DataDir)Pre-Processing_Results_$(data_name_prefix)
PreStore_suffix = / 
PreStore = $(patsubst $(trainfile_prefix)%$(trainfile_suffix),$(PreStore_prefix)%$(PreStore_suffix), $(trainfile))
# Network Models storage name
NetStore_prefix = $(DataDir)Network_Models_$(data_name_prefix)
NetStore_suffix = / 
NetStore = $(patsubst $(trainfile_prefix)%$(trainfile_suffix),$(NetStore_prefix)%$(NetStore_suffix), $(trainfile))
# training log name
trainlog_prefix = $(data_name_prefix)
trainlog_suffix = .log 
trainlog = $(patsubst $(trainfile_prefix)%$(trainfile_suffix),$(trainlog_prefix)%$(trainlog_suffix), $(trainfile))
# finish flag
# finish_pro = $(patsubst $(trainfile_prefix)%$(trainfile_suffix),.finish_pro%.swp, $(trainfile))


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
model:  $(NetStore)

$(PreStore_prefix)%$(PreStore_suffix)/.finshed : $(trainfile_prefix)%$(trainfile_suffix)
	python3 -u Data_Pre-Processing.py $< $(PreStore_prefix)$*$(PreStore_suffix) > $(trainlog_prefix)_Pre-$*$(trainlog_suffix) 2>&1
	touch $@

$(NetStore_prefix)%$(NetStore_suffix)/.finshed : $(PreStore_prefix)%$(PreStore_suffix) .Bulletin 
	python3 -u Data_Processing.py $@ $< $(NetStore_prefix)$*$(NetStore_suffix) > $(trainlog_prefix)_Train-$*$(trainlog_suffix) 2>&1
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
