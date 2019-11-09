import matplotlib.pyplot as plt
import numpy as np
import sys

filename = sys.argv[1]
output = sys.argv[2]
method = output.split('Distance')[0]

distanceDistri = np.load(filename)

tmax=max(distanceDistri)
tmin=min(distanceDistri)
average=np.mean(distanceDistri)
sigma=np.std(distanceDistri)

tbins=[i*0.5 for i in range(100)]

density = plt.hist(distanceDistri,tbins)[0]
plt.xlim([0,30])
plt.xlabel("distance")
plt.ylabel("entry number")
plt.title((method+" method:distance distribution"))
plt.text(20,int(max(density)/2),('$\mu={:.2f}$\n$\sigma={:.2f}$\nmax={:.2f}').format(average,sigma,tmax))
#plt.show()
plt.savefig(output)
