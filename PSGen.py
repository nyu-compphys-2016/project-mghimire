from __future__ import division
import numpy as np
import h5py as h5
import pylab as plt
import main

n = 100
L = 100
Om = 0.3
rhocrit = 8.58e-18
Mpchtokm = 4.081e19
kmtoMpch = 1/Mpchtokm
D0 = 0.0256745

initcon = h5.File("initcon.dat", "r")
results = h5.File("results.dat", "r")
Zeldovich = h5.File("Zeldovich.dat", "r")

I = initcon['array'][...]
R = results['array'][...]
Z = Zeldovich['array'][...]

initcon.close()
results.close()
Zeldovich.close()

data = open("linearpower_Box1000.dat", "r")

arr = []
for line in data:
    values = line.split()
    row = [float(x) for x in values]
    arr.append(row)
data.close()

P = np.array(arr)

meanrho = Om*rhocrit/kmtoMpch**3
Rho = main.CIC(R,L)
Zrho = main.CIC(Z,L)
##Irho = main.CIC(I,L)
delta = (Rho - meanrho)/meanrho
zelta = (Zrho - meanrho)/meanrho
##ielta = (Irho - meanrho)/meanrho

delk = np.fft.rfftn(delta)
zelk = np.fft.rfftn(zelta)
##ielk = np.fft.rfftn(ielta)

PR = np.zeros((n+1,3))  #first column is for frequencies, second column is for
                        #Power values, third column is for counting number of
                        #modes in k-shell
PZ = np.zeros((n+1,3))
##PI = np.zeros((n+1,3))

PR[:,0] = np.arange(n+1)*2*np.pi/L
PZ[:,0] = np.arange(n+1)*2*np.pi/L
##PI[:,0] = np.arange(n+1)*2*np.pi/L

for k in range(-n//2+1,n//2+1):
    for l in range(-n//2+1,n//2+1):
        for m in range(0,n//2+1):
            K2 = np.sqrt(k**2 + l**2 + m**2)
            PR[int(round(K2)),1] = PR[int(round(K2)),1] + np.abs(delta[k,l,m])**2
            PR[int(round(K2)),2] = PR[int(round(K2)),2] + 1
            PZ[int(round(K2)),1] = PZ[int(round(K2)),1] + np.abs(zelta[k,l,m])**2
            PZ[int(round(K2)),2] = PZ[int(round(K2)),2] + 1
            ##PI[int(round(K2)),1] = PI[int(round(K2)),1] + np.abs(ielta[k,l,m])**2
            ##PI[int(round(K2)),2] = PI[int(round(K2)),2] + 1


for i in range(n+1): #to get rid of division by zero
    if PR[i,2] == 0:
        PR[i,2] = 1
    if PZ[i,2] == 0:
        PZ[i,2] = 1
##    if PI[i,2] == 0:
##        PI[i,2] = 1

PR[:,1] = PR[:,1]/PR[:,2]
PZ[:,1] = PZ[:,1]/PZ[:,2]
##PI[:,1] = PI[:,1]/PI[:,2]


plt.bar(PR[:,0],PR[:,1],2*np.pi/100,color='mistyrose')
plt.bar(PZ[:,0],PZ[:,1],2*np.pi/100,color='palegreen')
plt.plot(P[:,0],P[:,1],"b-")
plt.xlabel("frequency $k$ $(h/$Mpc$)$")
plt.ylabel("Power $P(k)$ $((h/$Mpc$)^3)$")
plt.legend(prop={'size':10}, loc='upper right', labels=["Original", "Data", "Zel'dovich"])
plt.savefig("PSComparison")

##plt.bar(PI[:,0],PI[:,1],2*np.pi/100,color='mistyrose')
##plt.plot(P[:,0],P[:,1]*D0**2,"b-")
##plt.xlabel("frequency $k$ $(h/$Mpc$)$")
##plt.ylabel("Power $P(k)$ $((h/$Mpc$)^3)$")
##plt.legend(prop={'size':10}, loc='upper right', labels=["Original", "Data"])
##plt.savefig("IPSComparison")
