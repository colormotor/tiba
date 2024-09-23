'''
o-O-o o-O-o o--o    O
  |     |   |   |  / \
  |     |   O--o  o---o
  |     |   |   | |   |
  o   o-O-o o--o  o   o

Painting robot utils.
© Daniel Berio (@colormotor) 2023 - ...

todorov - Path-constrained minimum jerk trajectories

implementation of:
Todorov, E. and Jordan, M. (1998)
Smoothness maximization along a predefined path accurately predicts the speed profiles of
complex arm movements.

Pythpn port from Matlab implementation of Todorov here:
https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/58403/versions/7/previews/min_jerk_todorov.m/index.html
'''

from __future__ import division
import scipy
import numpy as np
import sys
import pdb, math
from . import utils
import nlopt

def print_same_line(s):
    sys.stdout.write("\r" + str(s))


colv = lambda v: np.array(v).reshape(-1, 1)
iter_count=[0]
iter_chars = ['-','/','-','\\']

def opt_nlopt(method):
    def subplex(fun, x0, maxiter):
        import nlopt
        opt = nlopt.opt(method, len(x0))
        opt.set_maxeval(5000) #7000) #maxiter)
        opt.set_min_objective(fun)
        opt.set_xtol_abs(1e-7) #10)
        opt.set_ftol_abs(1e-7) #10)
        opt.set_maxtime(6.0)
        x = opt.optimize(x0)
        return x
    return subplex


#function [trj, psg] = min_jerk_todorov(pos, dur, vel, acc, psg)
def min_jerk_todorov( pos, dur, vel=None, acc=None, psg=None, maxiter=5000, get_vel_acc=False, opt_method='NELDER_MEAD' ):
    N = pos.shape[0]
    D = pos.shape[1]

    # if not specified, bondary conditions are zero
    if vel is None:
        vel = np.zeros((2, D))
    if acc is None:
        acc = np.zeros((2, D))

    t0 = np.array([0, dur])

    methods = {
        'COBYLA': lambda f, x0, maxiter: scipy.optimize.minimize(f, x0, method='COBYLA', options={'maxiter':maxiter}).x,
        'BFGS': scipy.optimize.fmin_bfgs,
        'NELDER_MEAD': scipy.optimize.fmin,
        'POWELL': scipy.optimize.fmin_powell,
        'NLOPT_PRAXIS': opt_nlopt(nlopt.LN_PRAXIS),
        'NLOPT_BOBYQA': opt_nlopt(nlopt.LN_BOBYQA),
        'NLOPT_NELDERMEAD': opt_nlopt(nlopt.LN_NELDERMEAD),
        'NLOPT_SUBPLEX': opt_nlopt(nlopt.LN_SBPLX)
    }

    # optimize passage times if necessary
    print('Method:', opt_method)
    print('Dim:', D, 'vel:', vel, 'acc:', acc, 't0:', t0)
    if psg is None:
        if N < 2:
            raise ValueError("At least two points are required")
        if N==2:
            pos = np.array([pos[0], (pos[0]+pos[1])/2, pos[1]])
            N = 3
        if N > 2:
            print("Optimizing passage times for %d iterations"%maxiter)
            #psg = np.linspace(dur/(N-1), dur-dur/(N-1), N-2) # np.arange(dur/(N-1), dur-dur/(N-1), dur/(N-1)).reshape(-1, 1)
            psg = np.arange(dur/(N-1), dur-dur/(N-1)+1, dur/(N-1)).T
            if 'NLOPT' in opt_method:
                func = lambda psg_, grad: mjCOST(psg_, pos, vel, acc, t0)
            else:
                func = lambda psg_: mjCOST(psg_, pos, vel, acc, t0)
            ##%%OPTIONS.TolFun=10E-30; ftol
            ##%%OPTIONS.TolX =10E-30; xtol

            #%OPTIONS.MaxFunEvals=7500;
            #%OPTIONS.MaxIter=5000;
            print("optimizing\n")
            iter_count[0] = 0
            #psg = fmin(func=func, x0=psg)

            #fmin = scipy.optimize.fmin # Breaks with long trajectories
            #fmin = scipy.optimize.fmin_bfgs
            fmin = methods[opt_method] #scipy.optimize.fmin_cg
            #fmin =
            with utils.perf_timer('Fmin'):
                psg = fmin(func, psg,
                        maxiter=maxiter,
                        #maxfun=7500, #disp=True,
                        #ftol=10e-30,
                        #xtol=10e-30
                        ) #fminsearch(func, psg);

                psg = fmin(func, psg, maxiter=maxiter)
            print('done')
        else:
            psg = None
    trj = mjTRJ(psg, pos, vel, acc, t0, int(dur))
    if get_vel_acc:
        v, a = mjVelAcc(psg, pos, vel, acc, t0)
        return trj, psg, pos, v, a

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
    #T = np.vstack([T, T]).T
    T = np.vstack([T]*D).T


    xx0 = x[0:N-1,:]
    xx1 = x[1:N,:]


    j=(3*(3*aa0**2*T**4-2*aa0*aa1*T**4+3*aa1**2*T**4+24*aa0*T**3*vv0-
      16*aa1*T**3*vv0 + 64*T**2*vv0**2 + 16*aa0*T**3*vv1 -
      24*aa1*T**3*vv1 + 112*T**2*vv0*vv1 + 64*T**2*vv1**2 +
      40*aa0*T**2*xx0 - 40*aa1*T**2*xx0 + 240*T*vv0*xx0 +
      240*T*vv1*xx0 +240*xx0**2 - 40*aa0*T**2*xx1 + 40*aa1*T**2*xx1-
      240*T*vv0*xx1 - 240*T*vv1*xx1 - 480*xx0*xx1 + 240*xx1**2)/T**5)

    #pdb.set_trace()
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
        #T = np.vstack([T, T]).T #*np.ones((1,D))
        T = np.vstack([T]*D).T
        t = (t-tt[ii]) #*np.ones((1,D))
        t = np.vstack([t]*D).T
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


def mjVelAcc(t, x, v0, a0, t0):
    N = max(x.shape)
    D = min(x.shape)
    mat = np.zeros((2 * N - 4, 2 * N - 4))
    vec = np.zeros((2 * N - 4, D))
    #tt = np.concatenate((t0[0], t, t0[1]), axis=0)
    tt = np.concatenate([[t0[0]], t, [t0[1]]])

    for i in range(1, 2 * N - 4 + 1, 2):

        ii = int(math.ceil(i / 2.0))
        T0 = tt[ii] - tt[ii - 1]
        T1 = tt[ii + 1] - tt[ii]

        tmp = [-6 / T0, -48 / T0 ** 2, 18 * (1 / T0 + 1 / T1), \
               72 * (1 / T1 ** 2 - 1 / T0 ** 2), -6 / T1, 48 / T1 ** 2]

        if i == 1:
            le = 0
        else:
            le = -2

        if i == 2 * N - 5:
            ri = 1
        else:
            ri = 3

        mat[i - 1][i + le - 1:i + ri] = tmp[3 + le - 1:3 + ri]
        vec[i - 1][:] = 120 * (x[ii - 1][:] - x[ii][:]) / T0 ** 3 \
                        + 120 * (x[ii + 1][:] - x[ii][:]) / T1 ** 3

    for i in range(2, 2 * N - 4 + 1, 2):

        ii = int(math.ceil(i / 2.0))
        T0 = tt[ii] - tt[ii - 1]
        T1 = tt[ii + 1] - tt[ii]

        tmp = [48 / T0 ** 2, 336 / T0 ** 3, 72 * (1 / T1 ** 2 - 1 / T0 ** 2), \
               384 * (1 / T1 ** 3 + 1 / T0 ** 3), -48 / T1 ** 2, 336 / T1 ** 3]

        if i == 2:
            le = -1
        else:
            le = -3

        if i == 2 * N - 4:
            ri = 0
        else:
            ri = 2

        mat[i - 1][i + le - 1:i + ri] = tmp[4 + le - 1:4 + ri]
        vec[i - 1][:] = 720 * (x[ii][:] - x[ii - 1][:]) / T0 ** 4 \
                        + 720 * (x[ii + 1][:] - x[ii][:]) / T1 ** 4

    T0 = tt[1] - tt[0]
    T1 = tt[N - 1] - tt[N - 2]
    vec[0][:] = vec[0][:] + 6 / T0 * a0[0][:] + 48 / T0 ** 2 * v0[0][:]
    vec[1][:] = vec[1][:] - 48 / T0 ** 2 * a0[0][:] - 336 / T0 ** 3 * v0[0][:]
    vec[2 * N - 6][:] = vec[2 * N - 6][:] + 6 / T1 * a0[1][:] - 48 / T1 ** 2 * v0[1][:]
    vec[2 * N - 5][:] = vec[2 * N - 5][:] + 48 / T1 ** 2 * a0[1][:] - 336 / T1 ** 3 * v0[1][:]

    avav = np.linalg.inv(mat).dot(vec)
    a = avav[0:2 * N - 4:2][:]
    v = avav[1:2 * N - 4:2][:]

    return v, a

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%  compute intermediate velocities and accelerations
#function [v,a] = mjVelAcc(t, x, v0, a0, t0)
def mjVelAccOld(t, x, v0, a0, t0):
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
