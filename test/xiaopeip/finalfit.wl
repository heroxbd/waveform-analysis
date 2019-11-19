(* ::Package:: *)

BeginPackage["only`"];(*限定命名空间（上下文）*)
args=$ScriptCommandLine[[2]];numargs=ToExpression[args];(*读取命令行参数，并转化为数字类型*)
(*读入之前生成的单光子曲线和基准电压*)
winstring="./";
spe1=Import[winstring<>"medium/singlewave1.h5","spe"];
aver=Import[winstring<>"medium/average1.h5","averzero"];
spe2=Import[winstring<>"medium/singlewave2.h5","spe"];

defround[x_]:=Piecewise[{{x,x>0.1}},0];(*定义将小于0.1的数变为0，大于0.1的数不变的函数*)
opt={};(*初始化输出*)

For[j=1000*(numargs-1)+1,j<=Piecewise[{{1796736,numargs==1797}},1000*numargs],j++,(*根据命令行参数确定循环范围*)
(*开始循环，读入数据并将波形减去基准电压*)
ipt=Import[winstring<>"data/alpha-problem.h5","Waveform","TakeElements"->{j}][[1]];
wave=ipt["Waveform"]-972-aver;
event=ipt["EventID"];
channel=ipt["ChannelID"];

(*找到低于阈值电压（-6.5）的点lowp并掐头去尾*)
lowp=Flatten[Position[wave,x_/;x<-6.5]];
If[lowp!={},(*lowp为空则跳过后面的拟合步骤*)
If[lowp[[-1]]>1028,lowp=Drop[lowp,-1]];
If[lowp[[1]]<2,lowp=Drop[lowp,1]];

(*找到所有lowp周围（左7到右15）的点集，剔除越界的点，作为参与拟合的点nihep*)
nihep=Union@@Table[lowp+n,{n,-7,15}];
If[nihep[[-1]]>1029,nihep=Drop[nihep,-(nihep[[-1]]-1029)]];
If[nihep[[1]]<1,nihep=Drop[nihep,1-nihep[[1]]]];

(*找到二阶导大于1.5点向前平移9ns，并联合其左右的点作为寻根的集合possible*)
xuhao=Position[Table[wave[[i+1]]-wave[[i]]-wave[[i-1]]+wave[[i-2]],{i,lowp}],x_/;x>1.5]//Flatten;
possible=Union[lowp[[xuhao]]-10,lowp[[xuhao]]-9,lowp[[xuhao]]-8];

bianl=Table[b[y],{y,possible}];(*生成拟合系数b[n]，下标n在集合possible中*)
restr=VectorGreaterEqual[{bianl,0}];(*生成约束表达式b[n]>=0*)
mne=Table[spe1[[Piecewise[{{x-y+1,x-y+1>0}},x-y+1030]]],{x,nihep},{y,possible}];(*单光子曲线平移生成矩阵mne*)

(*求最小均方距离对应的系数b[n]，限定求解时间为0.25秒，将系数（小于0.1置为0）保存在ans中*)
ans=defround/@TimeConstrained[FindArgMin[{Norm[mne.bianl-wave[[nihep]]],restr},bianl],.25,
TimeConstrained[FindArgMin[{Norm[Table[spe2[[Piecewise[{{x-y+1,x-y+1>0}},x-y+1030]]],{x,nihep},{y,possible}].bianl-wave[[nihep]]],restr},bianl],.25]
],ans={}];

(*下面输出结果*)
If[Context[]!="only`",BeginPackage["only`"]];(*防止命名空间被篡改*)
If[AllTrue[ans,#<=0.05&],
	AppendTo[opt,{event,channel,FirstPosition[wave,Min[wave]][[1]]-9,1.}],(*如果没有在上面步骤中找到结果，则输出电压最小值对应的坐标*)
	For[k=1,k<=Length[ans],k++,
		If[ans[[k]]>0.05,
		AppendTo[opt,{event,channel,possible[[k]]-1,ans[[k]]}]]],(*将大于0.1的系数作为weight输出，对应的possible为PETime*)
	AppendTo[opt,{event,channel,FirstPosition[wave,Min[wave]][[1]]-9,1.}];
	];
 
If[Mod[j,100]==0,Print[j];Print[Context[]]];(*每完成100个波形进行一次打印*)
];(*结束循环*)

(*输出到文件*)
Export[winstring<>"result/"<>args<>"-pgan.h5",{"Answer"->opt},"Datasets"];
EndPackage[]

