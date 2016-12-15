from __future__ import division
import numpy as np
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

##XI = []
##YI = []
##ZI = []
##XR = []
##YR = []
##ZR = []
##XZ = []
##YZ = []
##ZZ = []
##
##for i in range(100**3):
##    if 98<I[i,2]<100:
##        XI.append(I[i,0])
##        YI.append(I[i,1])
##    if 98<R[i,2]<100:
##        XR.append(R[i,0])
##        YR.append(R[i,1])
##    if 98<Z[i,2]<100:
##        XZ.append(Z[i,0])
##        YZ.append(Z[i,1])
##
##XI = plt.array(XI)
##YI = plt.array(YI)
##XR = plt.array(XR)
##YR = plt.array(YR)
##XZ = plt.array(XZ)
##YZ = plt.array(YZ)
##
##plt.scatter(XI, YI, s=2, facecolor='0.5', lw = 0, c='b', marker='o')
##plt.xlabel('$x$ (Mpc/h)')
##plt.ylabel('$y$ (Mpc/h)')
##plt.savefig("init100")
##plt.clf()
##
##plt.scatter(XR, YR, s=2, facecolor='0.5', lw = 0, c='r', marker='o')
##plt.xlabel('$x$ (Mpc/h)')
##plt.ylabel('$y$ (Mpc/h)')
##plt.savefig("final100")
##plt.clf()
##
##plt.scatter(XZ, YZ, s=2, facecolor='0.5', lw = 0, c='g', marker='o')
##plt.xlabel('$x$ (Mpc/h)')
##plt.ylabel('$y$ (Mpc/h)')
##plt.savefig("Zeldovich100")
##plt.clf()
##
##plt.scatter(XZ, YZ, s=2, facecolor='0.5', lw = 0, c='g', marker='o')
##plt.scatter(XR, YR, s=2, facecolor='0.5', lw = 0, c='r', marker='o')
##plt.xlabel('$x$ (Mpc/h)')
##plt.ylabel('$y$ (Mpc/h)')
##plt.savefig("resultsoverlap100")
##plt.clf()

i = np.random.choice(np.arange(100**3),10000,replace=False)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
##ax.scatter(I[i,0],I[i,1],I[i,2], s=2, facecolor='0.5', lw = 0, c='b', marker='o')
ax.scatter(R[i,0],R[i,1],R[i,2], s=2, facecolor='0.5', lw = 0, c='r', marker='o')
##ax.scatter(Z[i,0],Z[i,1],Z[i,2], s=2, facecolor='0.5', lw = 0, c='g', marker='o')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
