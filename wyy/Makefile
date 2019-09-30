.PHONY: all

subfile = answer.h5
ansfile = /home/greatofdream/killxbq/training.h5
problemfile = /home/greatofdream/killxbq/training.h5
method = Delta

all: $(addprefix $(method),DistanceDistribution.png)

$(addprefix $(method),DistanceDistribution.png): $(addprefix $(method),DistanceDistribution.npy)
	python3 plotDistribution.py $^ $@

$(addprefix $(method),DistanceDistribution.npy): $(subfile) $(ansfile)
	python3 gradeMain.py $(method) $^ $@
$(subfile): $(problemfile)
	python3 calculate.py $^ $@

.DELETE_ON_ERROR:

.SECONDARY:
