# waveform-analysis

## Methods:pill:

+ tara DL
+ xuyu DL
+ gz246 EMMP
+ wyy delta
+ xiaopeip
+ xdcFT
+ lucyddm

## Frame:
The process of algorithm evaluation is automated in Makefile

For each method:
+ generate & save Answer of each training h5 file
+ record & save the efficiency of Answer generating
+ record & save average w&p-dist of each Answer respect to corresponding training h5 file

1. 测评程序的结果要求
   1. 生成的Answer中的每1条都需要计算w-dist和p-dist，计算结果独立保存成h5文件
   2. 记录每一个Answer(h5file)的平均w-dist和p-dist，保存成csv文件
   3. 画图的代码也包含在Makefile中
2. 协同合作要求
   1. 不直接修改Makefile
   2. 生成branch并对Makefile添加新代码
   3. 程序运行良好的情况下进行pull request