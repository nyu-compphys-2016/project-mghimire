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

#constants----------------------------------------------------------------------------------------
G = 6.67408e-20                         #gravitational constant in km^3/kgs^2
rhocrit = 8.58e-18                      #approximate critical density of universe in kg/km^3
Om = 0.3                                #density of matter today in units of rhocrit
Ode = 0.7                               #density of dark energy today in units of rhocrit
Mpcphtokm = 4.081e19                    #unit converters
kmtoMpcph = 1/Mpcphtokm         

#difeq function-----------------------------------------------------------------------------------
def f(r,a,L):
    "takes in r = [X,V], a, and L where X is an ordered vector of positions, V"
    "is an ordered vector of velocities, and a is the scale factor, and L is the"
    "box size, and returns the corresponding derivatives of X and V"
    X = r[:,0:3]%L
    V = r[:,3:6]
    derx = kmtoMpcph*V
    derv = -(3/(2*a))*(Ode*a**3/(Om+Ode*a**3)+1)*V-derphi(X,a,L)/dera(a)**2
    return np.concatenate((derx,derv),axis=1)

#auxiliary functions------------------------------------------------------------------------------
def dera(a):
    "takes in a and outputs derivative of a with respect to conformal time"
    return np.sqrt((8*np.pi*G*rhocrit/3)*(Om*a+Ode*a**4))

##derphi function---------------------------------------------------------------------------------
def derphi(X,a,L):
    "takes in vector of positions and scale factor and outputs the gradient of"
    "phi"
    Lk = L*Mpcphtokm
    meanrho = Om*rhocrit
    delta = (CIC(X,L)/Mpcphtokm**3 - meanrho)/meanrho
    delk = np.fft.rfftn(delta)
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

    return ICIC(X,xderphi,yderphi,zderphi,L)

##CIC function------------------------------------------------------------------------------------
def CIC(X,L):
    "takes in vector of positions and array of masses and outputs 3D array of"
    "densities at the grid centers"
    n = int(np.round(X.shape[0]**(1/3)))
    M = Om*rhocrit*L**3*Mpcphtokm**3/n**3   #mass per particle for evenly distributed mass in kg
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
    return rho

##ICIC function-----------------------------------------------------------------------------------
def ICIC(X,xderphi,yderphi,zderphi):
    "takes in vector of positions and 3D arrays of derphi in each direction"
    "and uses inverse cloud in cell interpolation to calculate effect of the"
    "derphi arrays on each point, and returns ordered vector of gradphi for"
    "each particle"
    n = xderphi.shape[0]
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

##init function-----------------------------------------------------------------------------------
def init(P,n,L):
    "takes in power spectrum and generates initial perturbations"
    A = np.zeros((n,n,n//2+1))
    B = np.zeros((n,n,n//2+1))
    for k in range(-n//2+1,n//2+1):
        for l in range(-n//2+1,n//2+1):
            for m in range(0,n//2+1):
                Pklm = Pw(P,L,k,l,m)
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
def Pw(P,L,k,l,m):
    "takes in power function and fourier coordinates and linearly interpolates power value for"
    "given coordinates"
    K = np.sqrt(k**2+l**2+m**2)
    K = K*(Pw.shape[0]-1)/L
    Power = P[int(K),1] + (K - int(K))*(P[int(K)+1,1]-P[int(K),1])
    return Power
