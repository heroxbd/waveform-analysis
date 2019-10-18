import grader
import analyzeH5 as an5
import sys
#import importlib
# importlib.reload(an5)
method = sys.argv[1]
sub = sys.argv[2]
ans = sys.argv[3]
N=-1

grader.grader(ans,sub,N,method)