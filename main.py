from __future__ import division
import numpy as np
import difeq
import h5py as h5
import pylab as plt

#contents-------------------------------------------------------------------------

##constants
###physical constants
###simulation parameters

##f: differential equation function (equations of motion)
##auxiliary functions to the differential equation function
###derphi: function that produces the gradient of phi
###CIC: cloud-in-cell interpolation function
###ICIC: inverse-cloud-in-cell interpolation function

##initial condition parameters
##init: initial position and velocity generator

##part that runs program and saves results

#constants----------------------------------------------------------------------------------------
G = 6.67408e-20                         #gravitational constant in km^3/kgs^2
rhocrit = 8.58e-18                      #approximate critical density of universe in kg/km^3
Om = 0.3                                #density of matter today in units of rhocrit
Ode = 0.7                               #density of dark energy today in units of rhocrit
Mpcphtokm = 4.081e19                    #unit converters
kmtoMpcph = 1/Mpcphtokm         

#simulation parameters----------------------------------------------------------------------------
L = 100                                 #mesh size per dimension in Mpc/h
Lk = L*Mpcphtokm                        #mesh size per dimension in km
n = 100                                 #number of particles per dimension
M = Om*rhocrit*L**3*Mpcphtokm**3/n**3   #mass per particle for evenly distributed mass in kg
N = 100                                 #number of steps for evolution of a
af = 1                                  #final a (today)

#difeq function-----------------------------------------------------------------------------------
def f(r,a):
    "takes in r = [X,V] and a, where X is an ordered vector of positions, V"
    "is an ordered vector of velocities, and a is the scale factor, and returns"
    "the corresponding derivatives of X and V"
    X = r[:,0:3]%L
    V = r[:,3:6]
    derx = kmtoMpcph*V
    derv = -(3/(2*a))*(Ode*a**3/(Om+Ode*a**3)+1)*V-derphi(X,a)/dera(a)**2
    return np.concatenate((derx,derv),axis=1)

#auxiliary functions------------------------------------------------------------------------------
def dera(a):
    "takes in a and outputs derivative of a with respect to conformal time"
    return np.sqrt((8*np.pi*G*rhocrit/3)*(Om*a+Ode*a**4))

##derphi function---------------------------------------------------------------------------------
def derphi(X,a):
    "takes in vector of positions and scale factor and outputs the gradient of"
    "phi"
    meanrho = Om*rhocrit
    delta = (CIC(X) - meanrho)/meanrho
    delk = np.fft.rfftn(delta)
##    phik = np.zeros(rhok.shape)
##    for k in range(rhok.shape[0]):
##        for l in range(rhok.shape[1]):
##            for m in range(rhol.shape[2]):
##                if not(k == l == m == 0):
##                    phik[k,l,m] = G*Om*rhocrit*Lk**2*rhok[k,l,m]\
##                                  /(np.pi*a*(k**2+l**2+m**2))
##    phi = np.fft.ifftn(phik)
    xderphik = np.zeros(delk.shape) + 0j
    yderphik = np.zeros(delk.shape) + 0j
    zderphik = np.zeros(delk.shape) + 0j
    for k in range(delk.shape[0]):
        for l in range(delk.shape[1]):
            for m in range(delk.shape[2]):
                K2 = k**2+l**2+m**2
                if not(K2 == 0):
                    xderphik[k,l,m] = -2j*G*Om*rhocrit*Lk*k*delk[k,l,m]/(a*K2)
                    yderphik[k,l,m] = -2j*G*Om*rhocrit*Lk*l*delk[k,l,m]/(a*K2)
                    zderphik[k,l,m] = -2j*G*Om*rhocrit*Lk*m*delk[k,l,m]/(a*K2)
    xderphi = np.fft.irfftn(xderphik)
    yderphi = np.fft.irfftn(yderphik)
    zderphi = np.fft.irfftn(zderphik)

    return ICIC(X,xderphi,yderphi,zderphi)

##CIC function------------------------------------------------------------------------------------
def CIC(X):
    "takes in vector of positions and array of masses and outputs 3D array of"
    "densities at the grid centers"
    rho = np.zeros([n,n,n])
    X = X/(L/n)
    Kp = X.astype(np.int64)
    K = Kp%n
    K1 = (Kp+1)%n
    D = X - Kp
    T = 1 - D
    for i in range(n**3):
        rho[K[i,0],K[i,1],K[i,2]] = rho[K[i,0],K[i,1],K[i,2]] + M*T[i,0]*T[i,1]*T[i,2]
        rho[K1[i,0],K[i,1],K[i,2]] = rho[K1[i,0],K[i,1],K[i,2]] + M*D[i,0]*T[i,1]*T[i,2]
        rho[K[i,0],K1[i,1],K[i,2]] = rho[K[i,0],K1[i,1],K[i,2]] + M*T[i,0]*D[i,1]*T[i,2]
        rho[K[i,0],K[i,1],K1[i,2]] = rho[K[i,0],K[i,1],K1[i,2]] + M*T[i,0]*T[i,1]*D[i,2]
        rho[K1[i,0],K1[i,1],K[i,2]] = rho[K1[i,0],K1[i,1],K[i,2]] + M*D[i,0]*D[i,1]*T[i,2]
        rho[K[i,0],K1[i,1],K1[i,2]] = rho[K[i,0],K1[i,1],K1[i,2]] + M*T[i,0]*D[i,1]*D[i,2]
        rho[K1[i,0],K[i,1],K1[i,2]] = rho[K1[i,0],K[i,1],K1[i,2]] + M*D[i,0]*T[i,1]*D[i,2]
        rho[K1[i,0],K1[i,1],K1[i,2]] = rho[K1[i,0],K1[i,1],K1[i,2]] + M*D[i,0]*D[i,1]*D[i,2]
    return rho/Mpcphtokm**3

##ICIC function-----------------------------------------------------------------------------------
def ICIC(X,xderphi,yderphi,zderphi):
    "takes in vector of positions and 3D arrays of derphi in each direction"
    "and uses inverse cloud in cell interpolation to calculate effect of the"
    "derphi arrays on each point, and returns ordered vector of gradphi for"
    "each particle"
    gradphi = np.zeros((n**3,3))
    X = X/(L/n)
    Kp = X.astype(np.int64)
    K = Kp%n
    K1 = (Kp+1)%n
    D = X - Kp
    T = 1 - D
    gradphi[:,0] = xderphi[K[:,0],K[:,1],K[:,2]]*T[:,0]*T[:,1]*T[:,2]+\
                   xderphi[K1[:,0],K[:,1],K[:,2]]*D[:,0]*T[:,1]*T[:,2]+\
                   xderphi[K[:,0],K1[:,1],K[:,2]]*T[:,0]*D[:,1]*T[:,2]+\
                   xderphi[K[:,0],K[:,1],K1[:,2]]*T[:,0]*T[:,1]*D[:,2]+\
                   xderphi[K1[:,0],K1[:,1],K[:,2]]*D[:,0]*D[:,1]*T[:,2]+\
                   xderphi[K[:,0],K1[:,1],K1[:,2]]*T[:,0]*D[:,1]*D[:,2]+\
                   xderphi[K1[:,0],K[:,1],K1[:,2]]*D[:,0]*T[:,1]*D[:,2]+\
                   xderphi[K1[:,0],K1[:,1],K1[:,2]]*D[:,0]*D[:,1]*D[:,2]
    gradphi[:,1] = yderphi[K[:,0],K[:,1],K[:,2]]*T[:,0]*T[:,1]*T[:,2]+\
                   yderphi[K1[:,0],K[:,1],K[:,2]]*D[:,0]*T[:,1]*T[:,2]+\
                   yderphi[K[:,0],K1[:,1],K[:,2]]*T[:,0]*D[:,1]*T[:,2]+\
                   yderphi[K[:,0],K[:,1],K1[:,2]]*T[:,0]*T[:,1]*D[:,2]+\
                   yderphi[K1[:,0],K1[:,1],K[:,2]]*D[:,0]*D[:,1]*T[:,2]+\
                   yderphi[K[:,0],K1[:,1],K1[:,2]]*T[:,0]*D[:,1]*D[:,2]+\
                   yderphi[K1[:,0],K[:,1],K1[:,2]]*D[:,0]*T[:,1]*D[:,2]+\
                   yderphi[K1[:,0],K1[:,1],K1[:,2]]*D[:,0]*D[:,1]*D[:,2]
    gradphi[:,2] = zderphi[K[:,0],K[:,1],K[:,2]]*T[:,0]*T[:,1]*T[:,2]+\
                   zderphi[K1[:,0],K[:,1],K[:,2]]*D[:,0]*T[:,1]*T[:,2]+\
                   zderphi[K[:,0],K1[:,1],K[:,2]]*T[:,0]*D[:,1]*T[:,2]+\
                   zderphi[K[:,0],K[:,1],K1[:,2]]*T[:,0]*T[:,1]*D[:,2]+\
                   zderphi[K1[:,0],K1[:,1],K[:,2]]*D[:,0]*D[:,1]*T[:,2]+\
                   zderphi[K[:,0],K1[:,1],K1[:,2]]*T[:,0]*D[:,1]*D[:,2]+\
                   zderphi[K1[:,0],K[:,1],K1[:,2]]*D[:,0]*T[:,1]*D[:,2]+\
                   zderphi[K1[:,0],K1[:,1],K1[:,2]]*D[:,0]*D[:,1]*D[:,2]

    return gradphi

##initial condition parameters--------------------------------------------------------------------
a0 = 0.02                               #scale factor for an initial redshift of 49
D0 = 0.0256745                          #linear growth function at redshift 49
da0 = dera(a0)                          #initial derivative of a with respect to conformal time
H0 = da0/a0                             #initial H

##init function-----------------------------------------------------------------------------------
def init(P):
    "takes in power spectrum and generates initial perturbations"
    A = np.zeros((n,n,n//2+1))
    B = np.zeros((n,n,n//2+1))
    for k in range(-n//2+1,n//2+1):
        for l in range(-n//2+1,n//2+1):
            for m in range(0,n//2+1):
                Pklm = Pw(P,k,l,m)
                A[k,l,m] = np.sqrt(Pklm/2)*np.random.normal(0,1)
                B[k,l,m] = np.sqrt(Pklm/2)*np.random.normal(0,1)
    delta = A+1j*B
    xphik = np.zeros((n,n,n//2+1)) + 0j
    yphik = np.zeros((n,n,n//2+1)) + 0j
    zphik = np.zeros((n,n,n//2+1)) + 0j
    for k in range(-n//2+1,n//2+1):
        for l in range(-n//2+1,n//2+1):
            for m in range(0,n//2+1):
                K2 = k**2+l**2+m**2
                if K2 != 0:
                    xphik[k,l,m] = -1j*k*delta[k,l,m]*L/(2*np.pi*K2)
                    yphik[k,l,m] = -1j*l*delta[k,l,m]*L/(2*np.pi*K2)
                    zphik[k,l,m] = -1j*m*delta[k,l,m]*L/(2*np.pi*K2)
    xphi = n**3*np.fft.irfftn(xphik)
    yphi = n**3*np.fft.irfftn(yphik)
    zphi = n**3*np.fft.irfftn(zphik)

    return xphi,yphi,zphi
##power interpolator------------------------------------------------------------------------------
def Pw(P,k,l,m):
    "takes in power function and fourier coordinates and linearly interpolates power value for"
    "given coordinates"
    K = np.sqrt(k**2+l**2+m**2)
    K = K*5000/L
    Power = P[int(K),1] + (K - int(K))*(P[int(K)+1,1]-P[int(K),1])
    return Power

##run functions-----------------------------------------------------------------------------------

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

Sx,Sy,Sz = init(P)

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

A,R = difeq.rk4(f,r0,a0,af,N)

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

