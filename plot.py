from __future__ import division
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py as h5

##reads results file and plots 3D plot of results

initcon = h5.File("initcon.dat", "r")
results = h5.File("results.dat", "r")
Zeldovich = h5.File("Zeldovich.dat", "r")

I = initcon['array'][...]
R = results['array'][...]
Z = Zeldovich['array'][...]

initcon.close()
results.close()
Zeldovich.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(I[:,0],I[:,1],I[:,2],c='b', marker='^')
ax.scatter(R[:,0],R[:,1],R[:,2],c='r', marker='o')
ax.scatter(Z[:,0],Z[:,1],Z[:,2],c='g', marker='o')

print I,R,I-R,R-Z

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
