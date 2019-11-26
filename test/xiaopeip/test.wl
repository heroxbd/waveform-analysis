numargs=0;
winstring="./";
spe1=Import[winstring<>"medium/singlewave1.h5","spe"];
aver=Import[winstring<>"medium/average1.h5","averzero"];
spe2=Import[winstring<>"medium/singlewave2.h5","spe"];
defround[x_]:=Piecewise[{{x,x>0.1}},0];
opt={};
j=1;
ipt=Import[winstring<>"data/alpha-problem.h5","Waveform","TakeElements"->{j}][[1]];

wave=ipt["Waveform"]-972-aver[[1]];
event=ipt["EventID"];
channel=ipt["ChannelID"];
lowp=Flatten[Position[wave,x_/;x<-6.5]];
nihep=Union@@Table[lowp+n,{n,-7,15}];
xuhao=Position[Table[wave[[i+1]]-wave[[i]]-wave[[i-1]]+wave[[i-2]],{i,lowp}],x_/;x>1.5]//Flatten;
possible=Union[lowp[[xuhao]]-10,lowp[[xuhao]]-9,lowp[[xuhao]]-8];
bianl=Table[b[y],{y,possible}];
mne=Table[spe1[[Piecewise[{{x-y+1,x-y+1>0}},x-y+1030]]],{x,nihep},{y,possible}];

defpos[x_]:=x>=0;
restr=AllTrue[bianl, defpos];
ans=defround/@FindArgMin[{Norm[mne.bianl-wave[[nihep]]],restr},bianl]
