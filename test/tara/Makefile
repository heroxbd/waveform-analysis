all : help

help:
	@cat ./mannual.txt

data_prefix = ztraining-
DataDir = ../../dataset/jinp

TrainDataDir = $(DataDir)
MaxSet = 10000000
NetDir = /srv/work_wuyy/Network_Models_ztraining-all
PreDir = $(NetDir)/PreProcess
NetStore_prefix = $(NetDir)/$(data_prefix)Channel
TrainData = $(wildcard $(TrainDataDir)/$(data_prefix)*.h5)
PreData = $(shell seq -f "$(NetDir)/PreProcess/Pre_Channel%g.h5" 0 29)
Training = $(patsubst $(NetDir)/PreProcess/Pre_Channel%.h5, $(NetStore_prefix)%/.Training_finished, $(PreData))
Nets = $(patsubst $(NetDir)/PreProcess/Pre_Channel%.h5, $(NetDir)/Nets/Channel%.torch_net, $(PreData))
PreTrained_Model = 

FileNo = *
ResultDir = ./
#test filename
testfile_prefix = $(DataDir)/$(data_prefix)
testfile = $(wildcard $(testfile_prefix)$(TestFileNo).h5)
# Result storage name
Result_prefix = $(ResultDir)Analyzing_Results_$(data_prefix)
Result = $(patsubst $(testfile_prefix)%.h5,$(Result_prefix)%/Prediction.h5, $(testfile))
W_dist_upper_limit=20


test : $(patsubst $(testfile_prefix)%.h5,$(Result_prefix)%/DistanceDistribution.pdf,$(testfile)) $(patsubst $(testfile_prefix)%.h5,$(Result_prefix)%/DistanceDistribution.csv,$(testfile))

Analysis : $(Result)

model : $(Nets)

.Bulletin:
	rm -f ./.bulletin.swp

$(Result_prefix)%/Prediction.h5 : $(testfile_prefix)%.h5 $(Nets) | .Bulletin
	@mkdir -p $(dir $@)
	python3 -u Prediction_Processing_Total.py $< $@ $(NetDir)/Nets -D 1 > $(dir $@)Analysis.log 2>&1

$(Result_prefix)%/DistanceDistribution.csv : $(Result_prefix)%/Record.h5
	python3 -u ../csv_dist.py $^ -o $@

$(Result_prefix)%/DistanceDistribution.pdf : $(Result_prefix)%/Record.h5
	python3 -u ../draw_dist.py $^ -o $@

$(Result_prefix)%/Record.h5 : $(Result_prefix)%/Prediction.h5 $(testfile_prefix)%.h5
	python3 -u ../test_dist.py $< --ref $(word 2,$^) -o $@

$(PreData) : PreProcess

PreProcess : $(TrainData)
	@mkdir -p $(PreDir)
	python3 -u Data_Pre-Processing.py $(TrainDataDir)/$(data_prefix) -o $(PreDir) -N $(MaxSet) > $(PreDir)/PreProcess.log 2>&1

$(NetStore_prefix)%/.Training_finished : $(NetDir)/PreProcess/Pre_Channel%.h5 | .Bulletin
	@mkdir -p $(dir $@)
	python3 -u Data_Processing.py $^ -n $* -B 64 -o $(dir $@) -P $(PreTrained_Model) > $(dir $@)Train.log  2>&1 
	@touch $@

$(NetDir)/Nets/Channel%.torch_net : $(NetStore_prefix)%/.Training_finished
	@mkdir -p $(dir $@)
	python3 -u Choose_Nets.py $(dir $<) $@

clean :

.DELETE_ON_ERROR: 

.SECONDARY:

.PHONY : all test clean

.PHONY : Analysis model .Bulletin

