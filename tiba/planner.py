import numpy as np
import matplotlib.pyplot as plt
from PyLQR.system import PosOrnPlannerSys,PosOrnKeypoint
from PyLQR.solver import ILQRRecursive
from PyLQR.sim import KDLRobot
from PyLQR.utils import primitives, PythonCallbackMessage
from polygonsoup import utils, plut, geom
from spatialmath import SE3, SO3, UnitQuaternion
from tqdm import tqdm, trange
from importlib import reload
import os, sys
import pdb
import numbers

from spatialmath.base import trnorm
import roboticstoolbox as rtb
from . import trajectory_utils as tu
from .config import config as cfg

reload(utils)
reload(tu)

def is_number(x):
    return isinstance(x, numbers.Number)

q_min_braccio = [0.0, 0.262, 0.0, 0.0, 0.0]
q_max_braccio = [np.pi, 2.879, np.pi, np.pi, np.pi]
braccio_limits = [q_min_braccio, q_max_braccio]

q_min_panda = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
q_max_panda = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])
panda_limits = [q_min_panda, q_max_panda]


def batch_indices(n, batch_size):
    n_batches = n//batch_size + 1
    if n_batches < 2:
        return [np.array(list(range(n)))]
    I = np.linspace(0, n-1, n_batches).astype(int)
    return [np.array(list(range(a, b+1))) for a, b in zip(I, I[1:])]

def check_limits(q):
    return np.all(q >= q_min_panda, axis=0) and np.all(q <= q_max_panda, axis=0)

class iLQRPlannerVelocity:
    def __init__(self, urdf_path, q0,
                 base_frame='tiba_link0',
                 tip_frame='tiba_pen_tcp'):
        self.q0 = q0
        self.dq0 = np.zeros(len(q0))
        self.rbt = KDLRobot(urdf_path, base_frame, tip_frame, self.q0, self.dq0)

    def plan(self, trajectories,
             num_iter=20,
             r=1e-5,
             weights=np.ones(7),
             rest = 0.0,
             limits=[]):
        if (is_number(trajectories[0][0])): # Single trajectory case
            trajectories = [trajectories]
        Q = np.diag(weights)
        trajectories = list(trajectories)
        # Add approaches between trajectories
        x0 = np.concatenate([self.rbt.get_ee_pos(), self.rbt.get_ee_orn()])
        linked_trajectories = []
        for traj in trajectories:
            if len(traj) == 0:
                continue
            approach = tu.gauss_approach(x0, traj[0])
            if approach:
                linked_trajectories.append(approach)
            if rest > 0.0:
                traj = tu.rest_inout(traj, rest)
            linked_trajectories.append(traj)
            x0 = traj[-1]

        # Plan
        nb_control_vars = len(self.dq0)
        Qs = []
        dQs = []
        u0_t = np.zeros(nb_control_vars)

        q0 = self.q0
        dq0 = self.dq0
        for tindex, states in enumerate(linked_trajectories):
            print('Planning trajectory %d of %d'%(tindex+1, len(linked_trajectories)))
            timesteps = list(range(len(states)))
            self.rbt.set_conf(q0, dq0, True)
            keypoints = [PosOrnKeypoint(states[ind][:3], states[ind][3:], Q, i) for i, ind in enumerate(timesteps)]
            horizon = len(timesteps)
            cmd_penalties = list(r*np.ones(nb_control_vars))

            if limits:
                sys = PosOrnPlannerSys(self.rbt, keypoints, cmd_penalties, limits[1], limits[0], horizon, 1, cfg.dt)
            else:
                sys = PosOrnPlannerSys(self.rbt, keypoints, cmd_penalties, horizon, 1, cfg.dt)

            planner = ILQRRecursive(sys)

            u0 = np.tile(u0_t, horizon-1)
            cb = PythonCallbackMessage()

            try:
                Qs_batch, _, dQs_batch, _, _ = planner.solve(
                    u0.reshape((-1, nb_control_vars)),
                    num_iter,
                    True, True, cb)

                Qs.append(Qs_batch)
                dQs.append(dQs_batch)
                q0 = Qs_batch[-1]
                dq0 = dQs_batch[-1]
                #u0_t = np.array(dQs_batch[-1])
            except Exception as e:
                print(e)

        print("Finished planning")
        return sum(Qs, []), sum(dQs, [])

def plan_trajectory(urdf_path,
                    q0,
                    trajectory,
                    batch_size=0,
                    num_iter=20,
                    init_phase_length=10,
                    base_frame='tiba_link0',
                    tip_frame='tiba_pen_tcp',
                    r=1e-5,
                    weights=np.ones(7),
                    limits=[]):
    # Load robot (assume zero velocity)
    dq0 = np.zeros(len(q0))
    rbt = KDLRobot(urdf_path, base_frame, tip_frame, q0, dq0)
    print(dir(rbt))
    # Weight matrix (assume fixed for now)
    Q = np.diag(weights)

    # Get initial end effector pose
    x0 = np.concatenate([rbt.get_ee_pos(), rbt.get_ee_orn()])
    states = []

    # Initial phase (try without interpolating orientation)
    for t in np.linspace(0, 1, init_phase_length):
        pos = x0[:3] + (trajectory[0][:3] - x0[:3])*t
        orn = x0[3:]
        states.append(np.concatenate([pos, orn]))

    # Append trajectory
    states = states + list(trajectory)

    if batch_size == 0:
        batch_size = len(states)
    else:
        print("Warning! Planning velocity commands for piecewise trajectory")
        print("Resulting speed may have discontinuities")

    inds = batch_indices(len(states), batch_size) #
    # num_batches = len(states) // batch_size
    # inds = np.linspace(0, len(states)-1, len(states)).astype(int)
    # if num_batches > 0:
    #     # Nb this will not necessarily result in the indicated batch size
    #     inds = np.array_split(inds, num_batches)
    # else:
    #     inds = [inds]
    nb_control_vars = len(dq0)
    Qs = []
    dQs = []
    u0_t = np.zeros(nb_control_vars)
    for I in inds:
        timesteps = list(I - I[0])
        rbt.set_conf(q0, dq0, True)
        keypoints = [PosOrnKeypoint(states[ind][:3], states[ind][3:], Q, i) for i, ind in enumerate(I)] #pose, t in zip(states[I], timesteps)]
        horizon = len(I)
        cmd_penalties = list(r*np.ones(nb_control_vars))

        # print(states[0][3:])
        # print(timesteps)
        # print(len(keypoints))
        print('creating system')
        if limits:
            sys = PosOrnPlannerSys(rbt, keypoints, cmd_penalties, limits[1], limits[0], horizon, 1, cfg.dt)
        else:
            sys = PosOrnPlannerSys(rbt, keypoints, cmd_penalties, horizon, 1, cfg.dt)

        print('created system')
        planner = ILQRRecursive(sys)


        u0 = np.tile(u0_t, horizon-1)
        cb = PythonCallbackMessage()
        print('Planning batch')
        try:
            Qs_batch, _, dQs_batch, _, _ = planner.solve(u0.reshape((-1, nb_control_vars)), num_iter, True, True, cb)
            print('Done')
            Qs.append(Qs_batch)
            #dQs_batch = np.cumsum(ddQs_batch, axis=0)
            dQs.append(dQs_batch)
            q0 = Qs_batch[-1]
            dq0 = dQs_batch[-1]
            u0_t = np.array(dQs_batch[-1])
        except Exception as e:
            print(e)

    # if not check_limits(Qs):
    #     print("Warning! Some joint limits were exceeded")
    #     print("Returning empty joint trajectory")
    #     return [], []

    return sum(Qs, []), sum(dQs, [])


def plan_trajectory_constant_orientation(urdf_path,
                                         from_pose,
                                         trajectory,
                                         orient,
                                         subd,
                                         q0,
                                         base_frame='tiba_link0',
                                         tip_frame='tiba_pen_tcp',
                                         angular_approach_step=0.03/10,
                                         pos_approach_step=1.0/1000,
                                         penalty=1e-5,
                                         niter=10,
                                         weights=np.ones(7),
                                         limits=[]):
    # Load robot
    rbt = KDLRobot(urdf_path, base_frame, tip_frame, q0, np.zeros(len(q0)))

    # Weight matrix
    Qtarget = np.diag(weights)

    # Find Euclidean and geodesic distance between source pose and first pose in trajectory
    first_pose = SE3(*trajectory[0]) * UnitQuaternion(orient).SE3()
    angdist = abs(from_pose.angdist(first_pose))
    dist = np.linalg.norm(from_pose.t - first_pose.t)
    # Compute approach steps based on distance
    ang_approach_steps = int(angdist/angular_approach_step)
    pos_approach_steps = int(dist/pos_approach_step)
    approach_steps = max(ang_approach_steps, pos_approach_steps)

    # Compute appriach keypoints if necessary
    if approach_steps > 0:
        timesteps = [i for i in range(approach_steps)]
        approach = from_pose.interp(first_pose, approach_steps)
        keypoints = [PosOrnKeypoint(pose.t, np.array(UnitQuaternion(pose)), Qtarget, t) for pose, t in zip(approach, timesteps)]
    else:
        keypoints = []

    # Compute trajectory
    timesteps = list(np.cumsum([approach_steps] + [subd for _ in range(len(trajectory))]))
    keypoints += [PosOrnKeypoint(p, orient, Qtarget, t) for t, p in zip(timesteps, trajectory)]

    nb_ctrl_var = len(q0)
    cmd_penalties = list(penalty*np.ones(nb_ctrl_var))
    #horizon = timesteps[-1]+1
    horizon = min(timesteps[-1]+1, 200)
    if limits:
        sys = PosOrnPlannerSys(rbt, keypoints, cmd_penalties, limits[1], limits[0], horizon, 1, cfg.dt)
    else:
        sys = PosOrnPlannerSys(rbt, keypoints, cmd_penalties, horizon, 1, cfg.dt)

    planner = ILQRRecursive(sys)

    u0_t = np.zeros(nb_ctrl_var)
    u0 = np.tile(u0_t, horizon-1)
    cb = PythonCallbackMessage()

    X2,F_X2,U2,K2,k2 = planner.solve(u0.reshape((-1, nb_ctrl_var)), niter, True, True, cb)
    #print(U2)
    return X2, U2

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

class DrawingPad:
    def __init__(self, path='pad.pkl'):
        self.path = os.path.expanduser(path)
        self.frame = np.eye(4)
        self.offset = 0.0
        self.box = None
        self.points = []
        self.pos_offset = [0.0, 0.0]
        self.scale = [1.0, 1.0]

    def clear(self):
        self.points.clear()
        self.save()

    def add_point(self, p):
        p = np.array(p)
        for q in self.points:
            if np.linalg.norm(p - q) < 3e-3:
                print('Redundant point!')
                return
        self.points.append(p)
        self.save()

    def save(self):
        utils.save_pkl(self.points, self.path)

    def load(self):
        try:
            self.points = utils.load_pkl(self.path)
        except FileNotFoundError as e:
            print(e)

    def is_valid(self):
        return len(self.points) > 3 and self.box is not None

    def z(self):
        return self.frame[:3, 2]

    def update(self, P=None):
        # Compute frame
        if P is not None:
            self.points = P

        if len(self.points) < 4:
            return

        U, S, mu = principal_components(self.points)
        frame = np.eye(4)
        frame[:3,:3] = U
        frame[:3, 3] = mu
        frame[:3, :3] = orthonormalize_xy(U[:3,2], np.eye(3))
        self.frame = frame

        # Compute bounding box for sample points
        self.box = compute_frame_aligned_xy_box(self.points, self.frame)
        scaled_box = [np.array(self.box[0]), np.array(self.box[1])]
        size = geom.rect_size(self.box)
        scaled_box[0] += self.pos_offset*size
        scaled_box[1] = scaled_box[0] + self.scale*size
        self.scaled_box = scaled_box

    def get_pen_orientation(self, robot, q):
        z = self.frame[:3,2]
        tcp_frame = robot.pen_tcp_frame(q)
        tcp_frame = align_frame_z(tcp_frame, -z)
        return np.array(UnitQuaternion(SE3(tcp_frame)))

    def fit_to_box(self, P, padding=0):
        ''' Fits an input 2d trajectory to the given 2d box'''
        return fit_to_box_and_augment(P, self.scaled_box, padding)

    def fit_to_frame(self, P, clamp_z=True):
        ''' Fits a 3d trajectory assumed to lie on the XY plan, to the frame of the drawing pad'''
        P = np.array(P)
        if clamp_z:
            P[:,2] = np.maximum(P[:,2], 0.0)
        return fit_to_frame(P, self.frame, self.offset)

    def fit_trajectory(self, P, padding=100):
        P = self.fit_to_box(P, padding) #fit_to_box_and_augment(P, self.box, padding)
        return self.fit_to_frame(P)
        #return fit_to_frame_box(P, self.box, self.frame, padding, offset=self.offset)

    def draw(self, v3d):
        v3d.frame(self.frame, 0.5)
        if self.is_valid():
            v3d.color(0, 0.4, 1.0, 1.0)
            draw_frame_aligned_xy_box(v3d, self.box, self.frame)
            v3d.color(0, 0.4, 1.0, 0.5)
            draw_frame_aligned_xy_box(v3d, self.box, self.frame, self.offset)
        if len(self.points):
            v3d.color(1, 0, 0, 0.3)
            v3d.points(self.points, 12)
            v3d.color(1, 0, 0, 1)
            v3d.points(self.points, 4)

def principal_components(X, resolve_signs=False):
    mu = np.mean(X, axis=0)
    X = (X - mu).T @ (X - mu)
    U, S, Vt = np.linalg.svd(X.T @ X)
    idx = S.argsort()[::-1]
    return U[:,idx], S[idx], mu


# def orthonormalize_xy(U, A):
#     ''' Compute X and Y axes based on z of U and x, y of A'''
#     V = np.array(U)
#     V[:3,0] = np.cross(A[:,1], U[:,2], )
#     V[:3,1] = np.cross(U[:,2], A[:,0])
#     # Must make sure we don't get precision issues
#     # https://github.com/bdaiinstitute/spatialmath-python/issues/28
#     V[:3,:3] = trnorm(V[:3,:3])
#     return V

def orthonormalize_xy(z, A):
    ''' Compute X and Y axes based on z of U and x, y of A'''
    V = np.array(A)
    V[:3,2] = z
    V[:3,0] = np.cross(A[:,1], z)
    V[:3,1] = np.cross(z, A[:,0])
    # Must make sure we don't get precision issues
    # https://github.com/bdaiinstitute/spatialmath-python/issues/28
    V[:3,:3] = trnorm(V[:3,:3])
    return V

def frame_to_pos(frame):
    pose = SE3(frame)
    return np.concatenate([pose.t, np.array(UnitQuaternion(pose))])

def align_frame_z(frame, z):
    z = np.array(z)
    frame = np.array(frame)
    frame[:3,2] = z
    frame[:3,0] = np.cross(frame[:3,1], z)
    frame[:3,1] = np.cross(z, frame[:3,0])
    # Must make sure we don't get precision issues
    # https://github.com/bdaiinstitute/spatialmath-python/issues/28
    return trnorm(frame)

def compute_frame_aligned_xy_box(P, frame):
    # Assume we are projecting to XY
    Q = geom.affine_transform(np.linalg.inv(frame), P)
    box = geom.bounding_box(Q[:,:2])
    return box

def get_offset_frame(frame, offset):
    z = frame[:3, 2]
    oframe = np.array(frame)
    oframe[:3, 3] = frame[:3, 3] + z*offset
    return oframe

def draw_frame_aligned_xy_box(v3d, box, frame, offset=0):
    P = geom.close(np.array(geom.rect_corners(box)))
    P = np.hstack([P, np.zeros([P.shape[0], 1])])
    frame = get_offset_frame(frame, offset)
    P = geom.affine_transform(frame, P)
    v3d.polyline(P)


def fit_to_box_and_augment(P, box, input_padding=10, flip_x=False, flip_y=True):
    P = np.array(P)
    if flip_y:
        P[:,1] = -P[:,1]
    if flip_x:
        P[:,0] = -P[:,0]
    P = geom.affine_transform(geom.rot_2d(geom.radians(-90)), P)
    mat = geom.rect_in_rect_transform(geom.bounding_box(P, input_padding), box)
    P = geom.affine_transform(mat, P)
    P = np.hstack([P, np.zeros([P.shape[0], 1])])
    return P

def fit_to_frame(P, frame, offset=0):
    ''' Fit a polyline to a box projected to XY plane defined by a given frame'''
    frame = get_offset_frame(frame, offset)
    return geom.affine_transform(frame, P)
