.PHONY: all
# answer filename
ansfile = /home/greatofdream/killxbq/ftraining-1.h5
# problem filename
problemfile = /home/greatofdream/killxbq/ftraining-1.h5
# name of the directory will be made to store the result
fileStore = ftraining-1
# submission file will be generated
subfile = Prediction_Results/Prediction_Mod_ftraining-1.h5
# method name and will use in the filename prefix
method = EMMP

# training filename
training = /home/greatofdream/killxbq/ftraining-0.h5
# model name prefix
trainStore = ftraining-0

all: $(subfile) 
# $(fileStore)/$(method)DistanceDistribution.png

$(fileStore)/$(method)DistanceDistribution.png: $(fileStore)/$(method)DistanceDistribution.npy
	python3 plotDistribution.py $^ $@

$(fileStore)/$(method)DistanceDistribution.npy: $(subfile) $(ansfile)
	mkdir -p $(fileStore)
	python3 graderMain.py $(fileStore)/$(method) $^ $@

$(subfile): MyAnswer.txt
	python3 write_txt2h5.py
MyAnswer.txt: $(problemfile)
	mkdir -p Prediction_Results
	matlab -nojvm -r "func_EMMP_SignalDet('$(problemfile)', 'MyAnswer.txt', 0, '$(training)')"
	# python3 write_txt2h5.py

