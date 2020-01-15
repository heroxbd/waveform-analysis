method:=fft
negative:=False
channelidb:=0
channelide:=30
eventid:=0
.PHONY: junoDataset junoAll junoAssess junoViewWave junoSpe jinpingAnswerView
junoDataDir:=dataset/juno
junoOutDir:=output/juno
junodataset:=$(wildcard $(junoDataDir)/hdf5/*/wave.h5)
junodst:=$(junodataset:$(junoDataDir)/hdf5/%/wave.h5=$(junoOutDir)/wave%/$(method)/answer.h5)
junodstpdf:=$(junodst:$(junoOutDir)/wave%/$(method)/answer.h5=$(junoOutDir)/wave%/$(method)/distanceDistribution.pdf)
junodstview:=$(junodst:$(junoOutDir)/wave%/$(method)/answer.h5=$(junoOutDir)/wave%/$(method)/figure/e$(eventid)c$(channelidb)-$(channelide).pdf)
junoOdstview:=$(junodst:$(junoOutDir)/wave%/$(method)/answer.h5=$(junoOutDir)/wave%/$(method)/figure/Origine$(eventid)c$(channelidb)-$(channelide).pdf)

junoDataset:
	mkdir -p $(junoDataDir)/hdf5
	cd $(junoDataDir)/hdf5 && ./scpJuno.sh

junoAll: $(junodst)
$(junoOutDir)/wave%/$(method)/answer.h5: $(junoDataDir)/hdf5/%/wave.h5 $(junoOutDir)/spe.h5
	mkdir -p $(dir $@)
	python3 src/main.py $^ $@ $(method) $(negative)

junoSpe: $(junoOutDir)/spe.h5
$(junoOutDir)/spe.h5: $(junodataset) 
	mkdir -p $(dir $@)
	python3 src/speGet.py  $^ -o $@ -n $(negative)

junoAssess: $(junodstpdf)
$(junoOutDir)/wave%/$(method)/distanceDistribution.pdf: $(junoOutDir)/wave%/$(method)/distanceDistribution.h5
	mkdir -p $(dir $@)
	python3 src/plotDistance.py $< -o $@
$(junoOutDir)/wave%/$(method)/distanceDistribution.h5: $(junoOutDir)/wave%/$(method)/answer.h5 $(junoDataDir)/hdf5/%/wave.h5 
	mkdir -p $(dir $@)
	python3 src/distance.py $< --ref $(word 2, $^) -o $(dir $@) 

junoAssessAll: $(junoOutDir)/waveAll/$(method)/distanceDistribution.pdf
$(junoOutDir)/waveAll/$(method)/distanceDistribution.pdf: $(junoOutDir)/waveAll/$(method)/distanceDistribution.h5
	mkdir -p $(dir $@)
	python3 src/plotDistance.py $< -o $@
$(junoOutDir)/waveAll/$(method)/distanceDistribution.h5: $(junodst:$(junoOutDir)/wave%/$(method)/answer.h5=$(junoOutDir)/wave%/$(method)/distanceDistribution.h5)
	mkdir -p $(dir $@)
	python3 src/merge.py $^ -o $@

junoAnswerView: $(junodstview)
$(junoOutDir)/wave%/$(method)/figure/e$(eventid)c$(channelidb)-$(channelide).pdf: $(junoOutDir)/wave%/$(method)/answer.h5 $(junoDataDir)/hdf5/%/wave.h5 $(junoOutDir)/spe.h5
	mkdir -p $(dir $@)
	python3 src/plotAnswer.py $^ $(dir $@) $(eventid) $(channelidb) $(channelide)

junoOriginAnswerView: $(junoOdstview)
$(junoOutDir)/wave%/$(method)/figure/Origine$(eventid)c$(channelidb)-$(channelide).pdf: $(junoOutDir)/spe.h5 $(junoDataDir)/hdf5/%/wave.h5
#output/juno/ztraining-3/lucyddm/figure/e1c0.pdf: $(junoOutDir)/ztraining-3/$(method)/answer.h5 $(junoDataDir)/hdf5/ztraining-3.h5 $(junoOutDir)/spe.h5
	mkdir -p $(dir $@)
	python3 src/fftview.py $^ $(dir $@) $(eventid) $(channelidb) $(channelide)

.PHONY: jinpingDataset jinpingAll jinpingAssess jinpingViewWave jinpingSpe
jinpingDataDir:=dataset/jinping
jinpingOutDir:=output/jinping
jinpingdataset:=$(wildcard $(jinpingDataDir)/hdf5/ztraining-*.h5)
jinpingdst:=$(jinpingdataset:$(jinpingDataDir)/hdf5/ztraining-%.h5=$(jinpingOutDir)/ztraining-%/$(method)/answer.h5)
jinpingdstpdf:=$(jinpingdst:answer.h5=distanceDistribution.pdf)
jinpingdstview:=$(jinpingdst:$(jinpingOutDir)/ztraining%/$(method)/answer.h5=$(jinpingOutDir)/ztraining%/$(method)/figure/e$(eventid)c$(channelidb)-$(channelide).pdf)
jinpingOdstview:=$(jinpingdst:$(jinpingOutDir)/ztraining%/$(method)/answer.h5=$(jinpingOutDir)/ztraining%/$(method)/figure/Origine$(eventid)c$(channelidb)-$(channelide).pdf)

jinpingDataset:
	mkdir -p $(jinpingDataDir)/hdf5

jinpingAll: $(jinpingdst)
$(jinpingOutDir)/%/$(method)/answer.h5: $(jinpingDataDir)/hdf5/%.h5 $(jinpingOutDir)/spe.h5
	mkdir -p $(dir $@)
	python3 src/main.py $^ $@ $(method) $(negative)

jinpingSpe: $(jinpingOutDir)/spe.h5
$(jinpingOutDir)/spe.h5: $(jinpingDataDir)/hdf5/ztraining-0.h5 
	mkdir -p $(dir $@)
	python3 src/speGet.py $^ $@

jinpingAssess: $(jinpingdstpdf)

$(jinpingOutDir)/ztraining-%/$(method)/distanceDistribution.pdf: $(jinpingOutDir)/ztraining-%/$(method)/distanceDistribution.h5
	mkdir -p $(dir $@)
	python3 src/plotDistance.py $< -o $@
$(jinpingOutDir)/ztraining-%/$(method)/distanceDistribution.h5: $(jinpingOutDir)/ztraining-%/$(method)/answer.h5 $(jinpingDataDir)/hdf5/ztraining-%.h5
	mkdir -p $(dir $@)
	python3 src/distance.py $< --ref $(word 2, $^) -o $(dir $@)

jinpingAnswerView: $(jinpingdstview)
$(jinpingOutDir)/%/$(method)/figure/e$(eventid)c$(channelidb)-$(channelide).pdf: $(jinpingOutDir)/%/$(method)/answer.h5 $(jinpingDataDir)/hdf5/%.h5 $(jinpingOutDir)/spe.h5
#output/jinping/ztraining-3/lucyddm/figure/e1c0.pdf: $(jinpingOutDir)/ztraining-3/$(method)/answer.h5 $(jinpingDataDir)/hdf5/ztraining-3.h5 $(jinpingOutDir)/spe.h5
	mkdir -p $(dir $@)
	python3 src/plotAnswer.py $^ $(dir $@) $(eventid) $(channelidb) $(channelide)

jinpingOriginAnswerView: $(jinpingOdstview)
$(jinpingOutDir)/%/$(method)/figure/Origine$(eventid)c$(channelidb)-$(channelide).pdf: $(jinpingOutDir)/%/$(method)/answer.h5 $(jinpingDataDir)/hdf5/%.h5
#output/jinping/ztraining-3/lucyddm/figure/e1c0.pdf: $(jinpingOutDir)/ztraining-3/$(method)/answer.h5 $(jinpingDataDir)/hdf5/ztraining-3.h5 $(jinpingOutDir)/spe.h5
	mkdir -p $(dir $@)
	python3 src/lucyview.py $^ $(dir $@) $(eventid) $(channelidb) $(channelide)

.DELETE_ON_ERROR:

.SECONDARY:
