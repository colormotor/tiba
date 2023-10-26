#!/usr/bin/env python3
#%%
'''
iLQR applied to a planar manipulator for a viapoints task (recursive formulation to find a controller)

Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
Written by Sylvain Calinon <https://calinon.ch>

This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
License: MIT
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from roboticstoolbox import ET as E
from roboticstoolbox import Robot
from polygonsoup import plut, geom
import numpy as np
import matplotlib.pyplot as plt

from importlib import reload
import gdavid.strokes as st
from generative import genetic_glyph as glyph
from polygonsoup import simplify
reload(glyph)

def get_segments(e, q, alpha=0.3, start=0):
    rob = Robot(e)
    print(rob.urdf_string)
    T = rob.fkine_all(q)
    segments = rob.segments()
    # draw all the line segments for the noodle plot

    for i, segment in enumerate(segments):
        linkframes = []
        for link in segment:
            if link is None:
                linkframes.append(rob.base)
            else:
                linkframes.append(T[link.number])
        points = np.array([linkframe.t for linkframe in linkframes])
        plut.stroke(points[start:], 'k', alpha=alpha)
        plut.draw_markers(points[start:], 'k', markersize=1.5, alpha=alpha)
        # self.links[i].set_xdata(points[:,0])
        # self.links[i].set_ydata(points[:,1])
    return T[-1].t

def approach(e, q, p, n):
    t = np.linspace(0, 1, n)
    x0 = e.fkine(q).t[:2]
    return np.array([x0 + (p-x0)*tt for tt in t])


def glyph_string(n, spacing, npts=6):
    P = []
    for i in range(n):
        Q = glyph.make_glyph([spacing*i, 0.0], 1.0, npts, min_ratio=0.9)
        print(Q.shape)
        P.append(Q)
    P = np.vstack(P)
    P = simplify.dce(P, 1)
    # return P
    return st.smooth_stroke_lqt(P, order=3, opt_x0=True, sigma=0.5, isotropy=0.5, subd=4, subsampling=1, angle=geom.degrees(50))


#e = E.Rz() * E.tx(1) * E.Rz() * E.tx(1) * E.Rz() * E.tx()
e = E.Rz() * E.tx(1)
for i in range(5):
   e = e * E.Rz() * E.tx(1)

e = E.tx() * E.Rz(-np.pi/2) * E.tx(0.6)
for i in range(3):
    e = e * E.Rz() * E.tx(1)

e = E.tx() * E.Rz(-np.pi/2) * E.tx(0.6)
for i in range(5):
    e = e * E.Rz() * E.tx(0.5)

njoints = len(e.joints())
Xhat = np.array([[2.8, 1.5], [2.9, -1]])
Xhat = np.array([[np.cos(t), np.sin(t)] for t in np.linspace(0, 2*np.pi, 10)])*0.5 + [0, -1.9]
#np.random.seed(102) # star
#Xhat = glyph_string(6, 1.0)
#np.random.seed(142) # interesting to make it longer
#Xhat = glyph_string(6, 2.0)
np.random.seed(13422) # interesting to make it longer
Xhat = glyph_string(6, 4.5, npts=5)

Xhat = geom.transform_to_rect(Xhat, geom.make_centered_rect([0, -1.8], [2.3, 1.5]))
#Xhat[:,:2] += np.random.uniform(-0.5, 0.5, (Xhat.shape[0], 2))*1

x0 = np.concatenate([[0.5], np.random.uniform(-np.pi, np.pi, njoints-1)]) #      np.array([0.3, 0.2, 0.5, 2])

#Xhat = geom.bspline(100, Xhat)
Xapproach = approach(e, x0, Xhat[0], 30)
#Xhat = np.vstack([, Xhat])
plut.figure((5,5))

get_segments(e, x0, start=1)
plut.stroke(Xapproach, 'b')
plut.stroke(Xhat, 'r')
plut.show(axis=True)

#%%
# Helper functions
# ===============================

def residuals(e, Q, Xh):
    X = np.array([pose.t[:2] for pose in e.fkine(Q.T)])
    #f = (Xhat.T - X.T) #
    f = (X.T - Xh.T)*1
    return f

def f_reach(e, Q, Xh):
    f = residuals(e, Q, Xh)
    n = Xh.shape[0]
    njoints = len(e.joints())
    J = np.zeros([Xh.shape[1], njoints, n])
    for t in range(n):
        j = e.jacob0(Q[:,t])
        J[:,:,t] = j[:Xh.shape[1]]
    return f, J

def damped_pinv(J, l=1e-3):
    lambdaI = np.eye(J.shape[0])*l
    # this formulation allows for 0 lambda (Deo & Walker 93)
    return J.T @ np.linalg.inv( J@J.T + lambdaI )


def ilqt_planar(e, x0, Xh, subd=1, niter=50, dt=0.001, q=1e1, r=2e-4, u0=None):
    # Parameters
    njoints = len(e.joints())
    param = lambda: None # Lazy way to define an empty class in python
    param.dt = dt #1e-3 # Time step length
    param.nbData = int(len(Xh)*subd)  # Number of datapoints
    param.nbIter = niter # Maximum number of iterations for iLQR
    param.nbPoints = len(Xh) # Number of viapoints
    param.nbVarX = njoints # State space dimension (x1,x2,x3)
    param.nbVarU = njoints # Control space dimension (dx1,dx2,dx3)
    param.nbVarF = Xh.shape[1]
    param.q = q # Tracking weighting term
    param.r = r # Control weighting term

    # Main program
    # ===============================
    # Time occurrence of viapoints

    #tl = np.linspace(0, param.nbData-1, param.nbPoints).astype(int)
    tl = np.linspace(0, param.nbData, param.nbPoints+1)
    tl = np.rint(tl[1:]).astype(np.int64) - 1
    # # iLQR (recursive)
    # # ===============================

    # Transfer matrices (for linear system as single integrator)
    A = np.eye(param.nbVarX)
    B = np.eye(param.nbVarX, param.nbVarU) * param.dt
    Su0 = np.vstack([
            np.zeros([param.nbVarX, param.nbVarX*(param.nbData-1)]),
            np.tril(np.kron(np.ones([param.nbData-1, param.nbData-1]), np.eye(param.nbVarX) * param.dt))
    ])
    Sx0 = np.kron(np.ones(param.nbData), np.identity(param.nbVarX)).T

    du = np.zeros([param.nbVarU, param.nbData-1])
    utmp = np.zeros([param.nbVarU, param.nbData-1])
    xtmp = np.zeros([param.nbVarX, param.nbData])

    k = np.zeros([param.nbVarU, param.nbData-1])
    K = np.zeros([param.nbVarU, param.nbVarX, param.nbData-1])
    Luu = np.identity(param.nbVarU) * param.r # Command cost Hessian is constant

    #x0 = np.array([3*np.pi/4, -np.pi/2, -np.pi/4]) # Initial state
    if u0 is None:
        uref = np.zeros([param.nbVarU, param.nbData-1]) # Initial control command
    else:
        uref = np.array(u0)

    xref = Su0 @ uref.flatten('F') + Sx0 @ x0 # System evolution
    xref = xref.reshape([param.nbVarX, param.nbData], order='F')

    for i in range(100): # param.nbIter ):
        f, J = f_reach(e, xref[:,tl], Xh)  # Residuals and Jacobians
        Lu = uref * param.r
        Lx = np.zeros([param.nbVarX, param.nbData])
        Lxx = np.zeros([param.nbVarX, param.nbVarX, param.nbData])
        for t in range(len(tl)):
            Lx[:,tl[t]] = J[:,:,t].T @ f[:,t] * param.q
            Lxx[:,:,tl[t]] = J[:,:,t].T @ J[:,:,t] * param.q

        # Backward pass
        Vx = Lx[:,-1] # Initialization
        Vxx = Lxx[:,:,-1] # Initialization
        for t in range(param.nbData-2,-1,-1):
            Qx = Lx[:,t] + A.T @ Vx
            Qu = Lu[:,t] + B.T @ Vx
            Qxx = Lxx[:,:,t] + A.T @ Vxx @ A
            QuuInv = np.linalg.inv(Luu + B.T @ Vxx @ B)
            #QuuInv = damped_pinv(Luu + B.T @ Vxx @ B)

            Qux = B.T @ Vxx @ A
            k[:,t] = -QuuInv @ Qu # Update the feedforward terms
            K[:,:,t] = -QuuInv @ Qux # Update the gains
            Vx = Qx - Qux.T @ QuuInv @ Qu # Propagate gradients
            Vxx = Qxx - Qux.T @ QuuInv @ Qux # Propagate Hessians

        # Forward pass, including step size estimate (backtracking line search method)
        alpha = 0.5
        cost0 = np.linalg.norm(f.flatten('F'))**2 * param.q + np.linalg.norm(uref)**2 * param.r # Cost
        while True:
            xtmp[:,0] = x0
            for t in range(param.nbData-1):
                du[:,t] = alpha * k[:,t] + K[:,:,t] @ (xtmp[:,t] - xref[:,t])
                utmp[:,t] = uref[:,t] + du[:,t]
                xtmp[:,t+1] = A @ xtmp[:,t] + B @ utmp[:,t]

            ftmp = residuals(e, xtmp[:,tl], Xh)
            #ftmp,_ = f_reach(xtmp[:,tl], param) # Residuals
            cost = np.linalg.norm(ftmp.flatten('F'))**2 * param.q + np.linalg.norm(utmp)**2 * param.r # Cost
            if cost < cost0 or alpha < 1e-3:
                uref = utmp
                xref = Su0 @ uref.flatten('F') + Sx0 @ x0
                xref = xref.reshape([param.nbVarX, param.nbData], order='F')
                print("Iteration {}, cost: {}".format(i,cost))
                break
            alpha /=2
        if np.linalg.norm(alpha * du) < 1e-2: # Early stop condition
            print('Early stopping')
            break


    u = np.zeros([param.nbVarU, param.nbData-1])
    x = np.zeros([param.nbVarX, param.nbData])
    x[:,0] = x0
    print(len(x))
    for t in range(param.nbData-1):
        u[:,t] = uref[:,t] + K[:,:,t] @ (x[:,t] - xref[:,t])
        x[:,t+1] = A @ x[:,t] + B @ u[:,t] # System evolution
    return x, u

x, u = ilqt_planar(e, x0, Xapproach, niter=10, q=1e1, r=1e-8)
x, u = ilqt_planar(e, x[:,-1], Xhat, subd=2, niter=50, q=1e1, r=1.0*2e-4)


print(len(x))
plut.figure((10,10))
#x0 = np.array([0.3, 0.2, 2])
#get_segments(e, x0, start=1) #[:,i])
traj = []
for i in range(x.shape[1]):
    p = get_segments(e, x[:,i], start=1, alpha=0.1)
    traj.append(p)
#plut.stroke(Xhat, 'r')
plut.stroke(np.array(traj), '0')
plut.show()


#%%
# Simulate reproduction with perturbation
# ===============================

xn = np.array([.5, 0, 0]) # Simulated perturbation on the state
tn = round(param.nbData/3) # Time occurrence of perturbation
u = np.zeros([param.nbVarU, param.nbData-1])
x = np.zeros([param.nbVarX, param.nbData])
x[:,0] = x0
for t in range(param.nbData-1):
        if t==tn:
                x[:,t] = x[:,t] + xn # Simulated perturbation on the state
        u[:,t] = uref[:,t] + K[:,:,t] @ (x[:,t] - xref[:,t])
        x[:,t+1] = A @ x[:,t] + B @ u[:,t] # System evolution


# Plots
# ===============================

plt.figure()
plt.axis('off')
plt.gca().set_aspect('equal', adjustable='box')

# Get points of interest
f00 = fkin0(x[:,0], param)
f01 = fkin0(x[:,tl[0]], param)
f02 = fkin0(x[:,tl[1]], param)
f = fkin(x, param)

plt.plot(f00[0,:], f00[1,:], c='black', linewidth=5, alpha=.2)
plt.plot(f01[0,:], f01[1,:], c='black', linewidth=5, alpha=.4)
plt.plot(f02[0,:], f02[1,:], c='black', linewidth=5, alpha=.6)
plt.plot(f[0,:], f[1,:], c='black', marker='o', markevery=[0]+tl.tolist())
plt.plot(f[0,tn-1:tn+1], f[1,tn-1:tn+1], c='green', marker='o') #Perturbation

# Plot bounding box or viapoints
ax = plt.gca()
color_map = ['deepskyblue','darkorange']
for t in range(param.nbPoints):
        rect_origin = param.Mu[:2,t] - param.A[:,:,t] @ np.array(param.sz)
        rect_orn = param.Mu[-1,t]
        rect = patches.Rectangle(rect_origin, param.sz[0]*2, param.sz[1]*2, np.degrees(rect_orn), color=color_map[t])
        ax.add_patch(rect)
        #plt.scatter(param.Mu[0,t], param.Mu[1,t], s=100, marker='X', c=color_map[t])

plt.show()

#%%
reload(planner)

x, u = planner.ilqt_planar(e, x0, Xapproach, niter=10, q=1e1, r=1e-8)
x, u = planner.ilqt_planar(e, x[:,-1], Xhat, subd=2, niter=50, q=1e1, r=1.0*2e-4)
plut.figure((5,5))
traj = []
for i in range(x.shape[1]):
    p = draw.esegments_2d(e,  x[:,i], start=1, alpha=0.1)[-1]
    traj.append(p)
plut.stroke(np.array(traj), '0')
plut.show()
