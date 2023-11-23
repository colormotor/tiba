# Python port of path-constrained minimum jerk trajectories
# from Matlab implementation of Todorov/
# https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/58403/versions/7/previews/min_jerk_todorov.m/index.html
from __future__ import division
from scipy.optimize import fmin
import numpy as np
import sys

def print_same_line(s):
    sys.stdout.write("\r" + str(s))


colv = lambda v: np.array(v).reshape(-1, 1)
iter_count=[0]
iter_chars = ['-','/','-','\\']

#function [trj, psg] = min_jerk_todorov(pos, dur, vel, acc, psg)
def min_jerk_todorov( pos, dur, vel=None, acc=None, psg=None, maxiter=5000, get_vel_acc=False ): 
    N = pos.shape[0]
    D = pos.shape[1]

    if vel is None:
        vel = np.zeros((2, D))
    if acc is None:
        acc = np.zeros((2, D))

    t0 = np.array([0, dur])

    # optimize passage times if necessary
    if psg is None:
        if N > 2:
            psg = np.linspace(dur/(N-1), dur-dur/(N-1), N-2) # np.arange(dur/(N-1), dur-dur/(N-1), dur/(N-1)).reshape(-1, 1)
            func = lambda psg_: mjCOST(psg_, pos, vel, acc, t0)
            ##%%OPTIONS.TolFun=10E-30; ftol
            ##%%OPTIONS.TolX =10E-30; xtol

            #%OPTIONS.MaxFunEvals=7500;
            #%OPTIONS.MaxIter=5000;
            #psg = fmin(func, psg, maxiter=5000, maxfun=7500, disp=True, ftol=10e-30, xtol=10e-30) #fminsearch(func, psg);
            print("optimizing\n")
            iter_count[0] = 0
            psg = fmin(func, psg, maxiter=maxiter)
            print('done')
        else:
            psg = []
    trj = mjTRJ(psg, pos, vel, acc, t0, int(dur))
    if get_vel_acc:
        v, a = mjVelAcc(psg, pos, vel, acc, t0)
        return trj, psg, v, a
        
    return trj, psg

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%  compute jerk cost
# function J = mjCOST(t, x, v0, a0, t0)

def mjCOST(t, x, v0, a0, t0):
    print_same_line(iter_chars[iter_count[0]%4])
    iter_count[0] += 1
    
    sys.stdout.write('.')
    N = x.shape[0]
    D = x.shape[1]
    
    v,a = mjVelAcc(t, x, v0, a0, t0)
    aa = np.vstack([a0[0,:],a, a0[1,:]])
    aa0 = aa[0:N-1,:]
    aa1 = aa[1:N,:]

    vv = np.vstack([v0[0,:],v, v0[1,:]])
    vv0 = vv[0:N-1,:]
    vv1 = vv[1:N,:]

    tt = np.concatenate([[t0[0]],t,[t0[1]]])
    T = np.diff(tt) #(1,D))
    T = np.vstack([T, T]).T
    
    xx0 = x[0:N-1,:]
    xx1 = x[1:N,:]

    j=(3*(3*aa0**2*T**4-2*aa0*aa1*T**4+3*aa1**2*T**4+24*aa0*T**3*vv0-
      16*aa1*T**3*vv0 + 64*T**2*vv0**2 + 16*aa0*T**3*vv1 -           
      24*aa1*T**3*vv1 + 112*T**2*vv0*vv1 + 64*T**2*vv1**2 +          
      40*aa0*T**2*xx0 - 40*aa1*T**2*xx0 + 240*T*vv0*xx0 +            
      240*T*vv1*xx0 +240*xx0**2 - 40*aa0*T**2*xx1 + 40*aa1*T**2*xx1- 
      240*T*vv0*xx1 - 240*T*vv1*xx1 - 480*xx0*xx1 + 240*xx1**2)/T**5)

    J = np.sum(np.sum(np.abs(j)))
    return J



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%  compute trajectory
#function X = mjTRJ(tx, x, v0, a0, t0, P)
def mjTRJ(tx, x, v0, a0, t0, P):
    N = x.shape[0]
    D = x.shape[1]
    
    if tx is not None:
        v, a = mjVelAcc(tx, x, v0, a0, t0)

        aa = np.vstack([a0[0,:], a, a0[1,:]])
        vv = np.vstack([v0[0,:], v, v0[1,:]]) 
        tt = np.concatenate([[t0[0]], tx, [t0[1]]])
    else:
       aa = a0
       vv = v0 
       tt = t0

    X = np.zeros((P, D))
   
    ii = 0
    for i in range(P):
        t = float(i)/(P-1)*(t0[1]-t0[0]) + t0[0]
        if( t > tt[ii+1] ):
            ii = ii+1 
        T = (tt[ii+1]-tt[ii])
        T = np.vstack([T, T]).T #*np.ones((1,D)) 
        t = (t-tt[ii]) #*np.ones((1,D))
        t = np.vstack([t, t]).T
        aa0 = aa[ii,:]; aa1 = aa[ii+1,:]
        vv0 = vv[ii,:]; vv1 = vv[ii+1,:]
        xx0 =  x[ii,:]; xx1 = x[ii+1,:]

        X[i,:] = (
        aa0*t**2/2 + t*vv0 + xx0 + t**4*(3*aa0*T**2/2 - aa1*T**2 +      
        8*T*vv0 + 7*T*vv1 + 15*xx0 - 15*xx1)/T**4 +                     
        t**5*(-(aa0*T**2)/2 + aa1*T**2/2 - 3*T*vv0 - 3*T*vv1 - 6*xx0+ 
        6*xx1)/T**5 + t**3*(-3*aa0*T**2/2 + aa1*T**2/2 - 6*T*vv0 -    
        4*T*vv1 - 10*xx0 + 10*xx1)/T**3)

    return X


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%  compute intermediate velocities and accelerations
#function [v,a] = mjVelAcc(t, x, v0, a0, t0)
def mjVelAcc(t, x, v0, a0, t0):
    N = x.shape[0]
    D = x.shape[1]
    

    mat = np.zeros((2*N-4,2*N-4))
    vec = np.zeros((2*N-4,D))
    tt = np.concatenate([[t0[0]], t, [t0[1]]])

    I = np.array(range(0, 2*N-4, 2))
    II = np.ceil(I/2)+1
    I, II = I.astype(int), II.astype(int)
    
    for i, ii in zip(I, II):
    #for i in range(0, 2*N-4, 2): #=1:2:2*N-4
        T0 = tt[ii] - tt[ii-1] 
        T1 = tt[ii+1] - tt[ii]
        tmp = np.array([-6/T0, -48/T0**2, +18*(1/T0+1/T1), 
         +72*(1/T1**2-1/T0**2), -6/T1, +48/T1**2])
        if i==0:
            le = 0
        else:
            le = -2

        if i == 2*N-5-1:
            ri = 1
        else:
            ri = 3
        mat[i,i+le:i+ri+1] = tmp[3+le-1:3+ri]
        vec[i,:] = 120*(x[ii-1,:]-x[ii,:])/T0**3 + 120*(x[ii+1,:]-x[ii,:])/T1**3 
        #print np.r_[i+le:i+ri+1]
        #print np.r_[3+le-1:3+ri]
        #print '----'
    I2 = np.array(range(1, 2*N-4, 2)).astype(int)
    
    for i, ii in zip(I2, II): #range(1, 2*N-4, 2): #=1:2:2*N-4
        T0 = tt[ii] - tt[ii-1] 
        T1 = tt[ii+1] - tt[ii]

        tmp = np.array([48/T0**2,        336/T0**3,    +72*(1/T1**2-1/T0**2),
             +384*(1/T1**3+1/T0**3),   -48/T1**2,    +336/T1**3])
        if i==1:
            le = -1
        else:
            le = -3

        if i==2*N-4-1:
            ri = 0
        else:
            ri = 2
        
        mat[i,i+le:i+ri+1] = tmp[4+le-1:4+ri]
        vec[i,:] = 720.*(x[ii,:]-x[ii-1,:])/T0**4 + 720*(x[ii+1,:]-x[ii,:])/T1**4
        #print np.r_[i+le:i+ri+1]
        #print np.r_[4+le-1:4+ri]
        #print '----'

    T0 = tt[1]-tt[0]
    T1 = tt[N-1]-tt[N-2]
    vec[0,:] = vec[0,:] + 6/T0*a0[0,:] + 48/T0**2*v0[0,:]
    vec[1,:] = vec[1,:] - 48/T0**2*a0[0,:] - 336/T0**3*v0[0,:]
    vec[2*N-5-1,:] = vec[2*N-5-1,:] + 6/T1*a0[1,:] - 48/T1**2*v0[1,:]
    vec[2*N-4-1,:] = vec[2*N-4-1,:] + 48/T1**2*a0[1,:] - 336/T1**3*v0[1,:]

    avav = np.linalg.solve(mat,vec)
    a = avav[range(0, N*2-4, 2),:] 
    v = avav[range(1, N*2-4, 2),:]
    
    return v, a

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,6))
    plt.axis('equal')
    P = np.random.uniform(-10,10, (5, 2))
    X, psg = min_jerk_todorov(P, 200)
    plt.plot(X[:,0], X[:,1], 'k')
    plt.plot(P[:,0], P[:,1], 'ro')
    plt.show()
