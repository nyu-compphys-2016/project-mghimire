from __future__ import division
import numpy as np
import difeq
import h5py as h5
import pylab as plt
import main

#simulation parameters----------------------------------------------------------------------------
L = 100                                 #mesh size per dimension in Mpc/h
Lk = L*Mpcphtokm                        #mesh size per dimension in km
n = 100                                 #number of particles per dimension
N = 100                                 #number of steps for evolution of a
af = 1                                  #final a (today)

##initial condition parameters--------------------------------------------------------------------
a0 = 0.02                               #scale factor for an initial redshift of 49
D0 = 0.0256745                          #linear growth function at redshift 49

data = open("linearpower_Box1000.dat", "r")

arr = []
for line in data:
    values = line.split()
    row = [float(x) for x in values]
    arr.append(row)
data.close()

P = np.array(arr)
P = np.concatenate(([[0,0]],P))
P[:,1] = P[:,1]*D0**2

Sx,Sy,Sz = main.init(P,n)

x = np.zeros((n,n,n,3))
Z = np.zeros((n,n,n,3))
for k in range(n):
    for j in range(n):
        for i in range(n):
            x[i,j,k] = np.array([i*L/n,j*L/n,k*L/n])
            Z[i,j,k] = np.array([i*L/n,j*L/n,k*L/n])

x[:,:,:,0] = x[:,:,:,0] + D0*Sx
x[:,:,:,1] = x[:,:,:,1] + D0*Sy
x[:,:,:,2] = x[:,:,:,2] + D0*Sz

Z[:,:,:,0] = Z[:,:,:,0] + Sx
Z[:,:,:,1] = Z[:,:,:,1] + Sy
Z[:,:,:,2] = Z[:,:,:,2] + Sz


v = np.zeros((n,n,n,3))
v[:,:,:,0] = D0*Sx*Mpcphtokm/a0
v[:,:,:,1] = D0*Sy*Mpcphtokm/a0
v[:,:,:,2] = D0*Sz*Mpcphtokm/a0

X = []
V = []
Zel = []

for k in range(n):
    for j in range(n):
        for i in range(n):
            X.append(x[i,j,k])
            V.append(v[i,j,k])
            Zel.append(Z[i,j,k])

X = np.array(X)%L

V = np.array(V)

Zel = np.array(Zel)%L

r0 = np.concatenate((X,V),axis=1)

def func(r,a):
    return main.f(r,a,L)

A,R = difeq.rk4(func,r0,a0,af,N)

IC = h5.File("initcon.dat", "w")
results = h5.File("results.dat", "w")
Zeldovich = h5.File("Zeldovich.dat", "w")

iset = IC.create_dataset("array", R[0][:,0:3].shape, dtype=np.float)
iset[...] = R[0][:,0:3]
dset = results.create_dataset("array", R[-1][:,0:3].shape, dtype=np.float)
dset[...] = R[-1][:,0:3]%L
pset = Zeldovich.create_dataset("array", Zel.shape, dtype=np.float)
pset[...] = Zel

IC.close()
results.close()
Zeldovich.close()
