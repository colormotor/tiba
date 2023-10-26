import numpy as np
import matplotlib.pyplot as plt
from PyLQR.system import PosOrnPlannerSys,PosOrnKeypoint
from PyLQR.solver import ILQRRecursive
from PyLQR.sim import KDLRobot
from PyLQR.utils import primitives, PythonCallbackMessage
from polygonsoup import utils, plut, geom
from tqdm import tqdm, trange
from importlib import reload
import sys

reload(utils)

q_min_braccio = [0.0, 0.262, 0.0, 0.0, 0.0]
q_max_braccio = [np.pi, 2.879, np.pi, np.pi, np.pi]
braccio_limits = [q_min_braccio, q_max_braccio]

q_min_panda = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
q_max_panda = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])
panda_limits = [q_min_panda, q_max_panda]

def plan_trajectory_constant_orientation(urdf_path,
                                         keypoints,
                                         orient,
                                         timesteps,
                                         dt,
                                         q0,
                                         base_frame,
                                         tip_frame,
                                         penalty=1e-5,
                                         niter=10,
                                         weights=np.ones(7),
                                         limits=[]):
    rbt = KDLRobot(urdf, base_frame, tip_frame, q0, np.zeros(len(q0)))
    if type(timesteps) == int:
        timesteps = [timesteps*i for i in range(len(keypoints))]

    Qtarget = np.diag(weights)
    keypoints = [PosOrnKeypoint(p, orient, Qtarget, t) for t, p in zip(timesteps, keypoints)]

    nb_ctrl_var = len(q0)
    cmd_penalties = list(penalty*np.ones(nb_ctrl_var))
    horizon = timesteps[-1]+1

    if limits:
        sys = PosOrnPlannerSys(rbt, keypoints, cmd_penalties, limits[1], limits[0], horizon, 1, dt)
    else:
        sys = PosOrnPlannerSys(rbt, keypoints, cmd_penalties, horizon, 1, dt)

    planner = ILQRRecursive(sys)

    u0_t = np.zeros(nb_ctrl_var)
    u0 = np.tile(u0_t, horizon-1)
    cb = PythonCallbackMessage()

    X2,F_X2,U2,K2,k2 = planner.solve(u0.reshape((-1,nb_ctrl_var)), niter, True, True, cb)
    return X2

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

def print_same_line(text):
    sys.stdout.write("\033[F")  # Move the cursor up one line
    sys.stdout.write("\033[K")  # Clear the line
    print(text)
    sys.stdout.flush()

def damped_pinv(J, l=1e-9):
    lambdaI = np.eye(J.shape[0])*l
    # this formulation allows for 0 lambda (Deo & Walker 93)
    return J.T @ np.linalg.inv( J@J.T + lambdaI )

def ilqr_planar(e, x0, Xh, subd=1, niter=50, dt=0.001, q=1e1, r=2e-4, u0=None):
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

    uref = np.zeros([param.nbVarU, param.nbData-1]) # Initial control command
    print(uref.shape)

    if u0 is not None:
        uref[:,0] = u0
        # print(u0.shape)
        # if u0.shape[1] < param.nbData-1:
        #     uref[:, u0.shape[1]] = u0
        # else:
        #     uref = u0[:, :param.nbData-1]
        # #uref = np.array(u0)

    xref = Su0 @ uref.flatten('F') + Sx0 @ x0 # System evolution
    xref = xref.reshape([param.nbVarX, param.nbData], order='F')

    tq = trange(param.nbIter, desc='Bar desc', leave=True)

    for i in tq: #tqdm(range(param.nbIter)):
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
            Quu = Luu + B.T @ Vxx @ B
            QuuInv = np.linalg.inv(Quu + np.eye(Quu.shape[0])*1e-6)
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
            if alpha < 1e-3 or cost < cost0:
                uref = utmp
                xref = Su0 @ uref.flatten('F') + Sx0 @ x0
                xref = xref.reshape([param.nbVarX, param.nbData], order='F')
                tq.set_description("Iteration {}, cost: {}".format(i,cost))
                tq.refresh() # to show immediately the update
                break
            alpha /=2
        if np.linalg.norm(alpha * du) < 1e-3: # Early stop condition
            print('Early stopping')
            break


    u = np.zeros([param.nbVarU, param.nbData-1])
    x = np.zeros([param.nbVarX, param.nbData])
    x[:,0] = x0

    for t in range(param.nbData-1):
        u[:,t] = uref[:,t] + K[:,:,t] @ (x[:,t] - xref[:,t])
        x[:,t+1] = A @ x[:,t] + B @ u[:,t] # System evolution
    print('Done')
    return x, u

ilqr_planar = ilqr_planar

def ilqr_planar_horizon(horizon, e, q0, Xh, subd=1, niter=50, dt=0.001, q=1e1, r=2e-4):
    n = len(Xh)
    nchunks = n // horizon
    if horizon < n:
        horizons = [[i*horizon, (i+1)*horizon] for i in range(nchunks)]
        horizons += [[horizons[-1][1], n]]
        #horizons = [horizons[0]] + [[h[0]-1, h[1]] for h in horizons[1:]]
    else:
        horizons = [[0, n]]

    u0 = None
    x = []
    us = []
    for a, b in horizons:
        xh, uh = ilqr_planar(e, q0, Xh[a:b], subd, niter, dt, q, r)
        u0 = uh[:,-1]
        us.append(uh)
        x += [xh]
        q0 = xh[:,-1]
    x = np.hstack(x)
    us = np.hstack(us)
    return x, us

def linear_approach_2d(e, q, p, n):
    t = np.linspace(0, 1, n)
    x0 = e.fkine(q).t[:2]
    return np.array([x0 + (p-x0)*tt for tt in t])
