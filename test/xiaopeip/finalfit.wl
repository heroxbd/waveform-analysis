(* ::Package:: *)

BeginPackage["only`"];
args=$ScriptCommandLine[[2]];numargs=ToExpression[args];

winstring="./";
spe1=Import[winstring<>"medium/singlewave1.h5","spe"];
aver=Import[winstring<>"medium/average1.h5","averzero"];
spe2=Import[winstring<>"medium/singlewave2.h5","spe"];

defround[x_]:=Piecewise[{{x,x>0.1}},0];
defpos[x_]:=x>=0;
opt={};

For[j=1000*(numargs-1)+1,j<=Piecewise[{{1796736,numargs==1797}},1000*numargs],j++,

ipt=Import[winstring<>"data/alpha-problem.h5","Waveform","TakeElements"->{j}][[1]];
wave=ipt["Waveform"]-972-aver[[1]];
event=ipt["EventID"];
channel=ipt["ChannelID"];

lowp=Flatten[Position[wave,x_/;x<-6.5]];
If[lowp!={},
If[lowp[[-1]]>1028,lowp=Drop[lowp,-1]];
If[lowp[[1]]<2,lowp=Drop[lowp,1]];

nihep=Union@@Table[lowp+n,{n,-7,15}];
If[nihep[[-1]]>1029,nihep=Drop[nihep,-(nihep[[-1]]-1029)]];
If[nihep[[1]]<1,nihep=Drop[nihep,1-nihep[[1]]]];

xuhao=Position[Table[wave[[i+1]]-wave[[i]]-wave[[i-1]]+wave[[i-2]],{i,lowp}],x_/;x>1.5]//Flatten;
possible=Union[lowp[[xuhao]]-10,lowp[[xuhao]]-9,lowp[[xuhao]]-8];

bianl=Table[b[y],{y,possible}];
restr=AllTrue[bianl, defpos];
mne=Table[spe1[[Piecewise[{{x-y+1,x-y+1>0}},x-y+1030]]],{x,nihep},{y,possible}];

ans=defround/@TimeConstrained[FindArgMin[{Norm[mne.bianl-wave[[nihep]]],restr},bianl],.25,
TimeConstrained[FindArgMin[{Norm[Table[spe2[[Piecewise[{{x-y+1,x-y+1>0}},x-y+1030]]],{x,nihep},{y,possible}].bianl-wave[[nihep]]],restr},bianl],.25]
],ans={}];

If[Context[]!="only`",BeginPackage["only`"]];
If[AllTrue[ans,#<=0.05&],
	AppendTo[opt,{event,channel,FirstPosition[wave,Min[wave]][[1]]-9,1.}],
	For[k=1,k<=Length[ans],k++,
		If[ans[[k]]>0.05,
		AppendTo[opt,{event,channel,possible[[k]]-1,ans[[k]]}]]],
	AppendTo[opt,{event,channel,FirstPosition[wave,Min[wave]][[1]]-9,1.}];
	];
 
If[Mod[j,100]==0,Print[j];Print[Context[]]];
];

Export[winstring<>"result/"<>args<>"-pgan.h5",{"Answer"->opt},"Datasets"];
EndPackage[]
