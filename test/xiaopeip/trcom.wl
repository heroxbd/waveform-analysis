#!/usr/bin/env wolframscript

fileName=Last[$ScriptCommandLine];
in=Import[fileName,"Text"];
out=StringReplace[in,("\\:"~~(n:Repeated[HexadecimalCharacter,{4}])):>FromCharacterCode[FromDigits[n,16],"UTF-8"]];
Export[fileName,out,"Text",CharacterEncoding->"UTF-8"];