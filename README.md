# waveform-analysis

## Methods:pill:

+ tara DL
+ xuyu DL
+ gz246 EMMP
+ wyy delta
+ xiaopeip
+ xdc

## Frame:
The process of algorithm evaluation is automated in Makefile

For each method:
+ generate & save Answer of each training h5 file
+ record & save the efficiency of Answer generating
+ record & save average w&p-dist of each Answer respect to corresponding training h5 file

1. 测评程序的结果要求
   1. 生成的Answer中的每1条都需要计算w-dist和p-dist，计算结果独立保存成 h5file
   2. 记录每一个Answer(h5file)的平均w-dist和p-dist，保存成csv文件
   3. 画图的代码也写在Makefile中
2. 协同合作要求
   1. 不直接修改Makefile
   2. 生成branch并对Makefile添加新代码
   3. 程序运行良好的情况下进行pull request

## method

### LucyDDM

+ `make lucyddm` 运行jinping数据
+ `make lucyddmJuno`运行**JUNO**数据
+ `make junoDataset`下载**JUNO**数据
  
问题：
+ 不同spe产生方式对算法评测有影响，应使用同一个spe
+ 不同算法应放在一个库中，以函数调用执行，方便之后使用同一个框架与并行化
+ 数据和代码应该分离

问题的解决
+ 数据放在了dataset文件夹
+ spe_get.py和wf_analysis_func.py用于生成spe
