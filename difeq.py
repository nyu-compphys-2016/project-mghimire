from __future__ import division
import numpy as np

def euler(f,x0,t0,tn,N):
        "takes a function f for the first derivative of x, value x0 of x at t0,"
        "the last time tn of the interval, and the number of time steps N"
        "and outputs arrays t,x(t) for each time step using the Forward-Euler"
        "Euler"

        h = (tn - t0)/N
        
        t = np.arange(t0,tn+h,h)
        x = []
        for i in t:
                x.append(x0)
                x0 = x0 + h*f(x0,i)
        return t,x

def rk2(f,x0,t0,tn,N):
        "takes a function f for the first derivative of x, value x0 of x at t0,"
        "the last time tn of the interval, and the number of time steps N"
        "and outputs arrays t,x(t) for each time step using the second order"
        "Runge-Kutta method"

        h = (tn - t0)/N

        t = np.arange(t0,tn+h,h)
        x = []
        for i in t:
                x.append(x0)
                k1 = h*f(x0,i)
                k2 = h*f(x0+0.5*k1,i+0.5*h)
                x0 = x0 + k2
        return t,x

def rk4(f,x0,t0,tn,N):
        "takes a function f for the first derivative of x, value x0 of x at t0,"
        "the last time tn of the interval, and the number of time steps N"
        "and outputs arrays t,x(t) for each time step using the fourth order"
        "Runge-Kutta method"

        h = (tn - t0)/N

        t = np.arange(t0,tn+h,h)
        x = []
        for i in t:
                x.append(x0)
                k1 = h*f(x0,i)
                k2 = h*f(x0+0.5*k1,i+0.5*h)
                k3 = h*f(x0+0.5*k2,i+0.5*h)
                k4 = h*f(x0+k3,i+h)
                x0 = x0 + (k1+2*k2+2*k3+k4)/6
        return t,x

def verlet(f,R0,t0,tn,N):
        "takes a function f for the time derivatives of x, R0 = [x0,v0]"
        "at time t0, the last time tn of the interval, and the number of time"
        "steps N and outputs arrays t,x(t) for each time step using the Verlet"
        "method"

        m = int(R0.shape[0])    #the output of f and R0 have the same dimension

        h = (tn - t0)/N

        t = np.arange(t0,tn+h,h)  
        R = []
        x0 = R0[0:int(m/2)]
        v0 = R0[int(m/2):m]
        
        vhlf = v0 + 0.5*h*f(R0,t0)[int(m/2):m]
        for i in t:
                R.append(R0)
                x0 = x0 + h*vhlf
                R0 = np.array(list(x0) + list(v0))
                k = h*f(R0,i+h)[int(m/2):m]
                v0 = vhlf + 0.5*k
                vhlf = vhlf + k
                R0 = np.array(list(x0) + list(v0))
        return t,R

def rk4step(f,x,i,h):
        "does a single calculation of the next x1 using the fourth order"
        "Runge-Kutta method"
        k1 = h*f(x,i)
        k2 = h*f(x+0.5*k1,i+0.5*h)
        k3 = h*f(x+0.5*k2,i+0.5*h)
        k4 = h*f(x+k3,i+h)
        return x + (k1+2*k2+2*k3+k4)/6

def rk4ad(f,x0,t0,tn,ep):
        "takes a function f for the first derivative of x, value x0 of x at t0,"
        "the last time tn of the interval, and the error threshold ep"
        "and outputs arrays t,x(t) for each time step using an adaptive fourth"
        "order Runge-Kutta method"
        t = []
        x = []
        h = ep
        ti = 0.0
        x.append(x0)
        t.append(ti)
        while ti+2*h < tn:
                x1 = rk4step(f,x0,ti,h)
                x1 = rk4step(f,x1,ti+h,h)
                x2 = rk4step(f,x0,ti,2*h)
                if np.all(x1 == x2):
                        rho = 16.0
                else:
                        rho = 30*h*ep/np.sqrt(((x1-x2)**2).sum()) #taking norm
                if rho > 16.0:
                        rho = 16.0      #maxing out step change by factor of 2
                if rho >= 1:
                        ti = ti+2*h
                        h = h*rho**0.25
                        x0 = x1
                        x.append(x0)
                        t.append(ti)
                else:
                        h = h*rho**0.25
                if ti + 2*h >= tn:               #final step checker
                        x0 = rk4step(f,x0,ti,(tn-ti)/2)
                        x0 = rk4step(f,x0,ti+(tn-ti)/2,(tn-ti)/2)
                        ti = tn
                        x.append(x0)
                        t.append(ti)
        return t,x






        
