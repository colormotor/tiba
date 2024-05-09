'''
o-O-o o-O-o o--o    O
  |     |   |   |  / \
  |     |   O--o  o---o
  |     |   |   | |   |
  o   o-O-o o--o  o   o

Painting robot utils.
© Daniel Berio (@colormotor) 2023 - ...

trajectory_utils - Trajectory generation utilities
'''


from importlib import reload
import numpy as np
import networkx as nx
from . import geom, bezier, simplify
from itertools import combinations
from scipy.special import erf
from spatialmath import SE3, SO3, UnitQuaternion
from spatialmath.base import trnorm
from scipy.interpolate import interp1d, splprep, splev, make_interp_spline, CubicSpline, BPoly
from roboticstoolbox.tools.trajectory import ctraj, trapezoidal_func, trapezoidal
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from collections import defaultdict
from math import hypot
import trajectory_smoothing
reload(trajectory_smoothing)
from . import trapezoidal, todorov, lqt
reload(trapezoidal)
reload(todorov)


from .fitCurves.fitCurves import fitCurve

from . import config
from .config import cfg

import numbers
import pdb


def is_number(x):
    return isinstance(x, numbers.Number)

def to_pos_orn(pose):
    ''' Convert a pose to 7d vector: 3d position and quaternion orientation'''
    if (type(pose) == list or
        (type(pose) == np.ndarray and len(pose.shape) == 1)):
        return pose
        #return pose.t, np.array(UnitQuaternion(pose))
    pose = SE3(pose)
    pose = pose.norm()
    return np.concatenate([pose.t, np.array(UnitQuaternion(pose))])

def to_pose(pose):
    ''' Convert input (SE3 or matrix) to a SE3 pose'''
    if type(pose) == SE3:
        return pose.norm()
    if type(pose) == np.ndarray and len(pose.shape) == 2:
        return SE3(pose).norm()
    pose = SE3(*pose[:3])*(UnitQuaternion(pose[3:])).SE3()
    return pose.norm()

def to_frame(pose):
    return np.array(to_pose(pose))

def append_rest(traj, duration):
    ''' Append rest to trajectory for a given duration'''
    traj = list(traj)
    n = int(duration/cfg.dt)
    if not n:
        return traj
    if not traj:
        print('Cannot add rest to empty trajectory')
        return traj
    print('appending %d rest'%n)
    return traj + [np.array(traj[-1])]*n

def insert_rest(traj, duration):
    ''' Insert rest at beginning of trajectory for a given duration'''
    traj = list(traj)
    n = int(duration/cfg.dt)
    if not n:
        return traj
    if not traj:
        print('Cannot add rest to empty trajectory')
        return traj
    print('inserting %d rest'%n)
    return [np.array(traj[0])]*n + traj

def rest_inout(traj, duration):
    ''' Insert and append rest'''
    traj = list(traj)
    n = int(duration/cfg.dt)
    if not n:
        return traj
    return [np.array(traj[0])]*n + traj + [np.array(traj[-1])]*n

# Trajectory generators
def SE3_interp(a, b, t):
    ''' Spherical linear interpolation between two SE3 poses
    The spatialmath version (`a.slerp(b, t)`) seems to break for simlar rotations
    See: https://github.com/bdaiinstitute/spatialmath-python/issues/41
    Maybe due to numerical issues?
    '''
    pa = a.t
    pb = b.t
    qa = UnitQuaternion(a)
    qb = UnitQuaternion(b)
    rots = R.from_quat([np.array(qa), np.array(qb)])
    slerp = Slerp([0, 1], rots)
    d = pb - pa
    if type(t) is float:
        res = np.concatenate([pa + t*d, slerp(t).as_quat()])
        print(SE3.isvalid(np.array(to_pose(res))))
        return res
    res = [np.concatenate([pa + ti*d, slerp(ti).as_quat()]) for ti in t]
    return res


def offset_frame(frame, d):
    frame = np.array(frame)
    frame[:3, 3] = frame[:3, 3] + d
    return frame

def approach_inout(x, trajectory, kind='trapezoidal', **args):
    if kind=='linear':
        approach = linear_appoach
    elif kind=='trapezoidal':
        approach = trapezoidal_approach
    else:
        approach = gauss_approach

    return (approach(x, trajectory[0], **args) +
            list(trajectory) +
            approach(trajectory[-1], x, **args))

def joint_trajectory_time_optimal(Q_, max_speed=None, accel_time=None, tol=0, dt=None): #np.pi/2000.0):
    from trajectory_smoothing import TrajectorySmoother
    Q_ = np.array(Q_)
    Q = [Q_[0]]
    for q1, q2 in zip(Q_, Q_[1:]):
        d = np.linalg.norm(q2 - Q[-1])
        if d <= tol:
            print("Skipping joint distance: ", d)
            continue
        Q.append(q2)
    if len(Q) < 2:
        print(len(Q), len(Q_))
        print("Trajectory too short")
        return [], []

    Q = np.array(Q)

    if max_speed is None:
        max_speed = cfg.joint_speed
    if accel_time is None:
        accel_time = cfg.joint_acceleration_time
    max_accel = max_speed / accel_time

    dtype = np.float64

    dof = len(Q[0])
    # Max velocity and acceleration combined along unit vector
    # Note that otherwise we would get higher max speed and acceleration maginitude
    max_velocity = np.ones(dof, dtype=dtype) * max_speed
    max_acceleration = np.ones(dof, dtype=dtype) * max_accel

    print("Smoother")
    smoother = TrajectorySmoother(
        dof, max_velocity, max_acceleration, tol
    )

    if dt is None:
        dt = cfg.robot_dt
    Q = np.array(Q).astype(dtype)
    result = smoother.smooth_interpolate(Q, interpolation_dt=dt, traj_dt=0.001)
    if not result.success:
        print("Failed to compute joint trajectory")

    Q = np.array(result.position, dtype=dtype)
    #dQ = np.vstack([np.zeros_like(Q[0]), np.diff(Q, axis=0)/cfg.robot_dt])
    dQ = np.array(result.velocity, dtype=dtype)
    # Looks like the planner does not create zero velocity at the end?
    # dQ = np.vstack([dQ] + [np.zeros_like(dQ[0])]*500)
    # Q = np.vstack([Q] + [Q[-1]]*500)

    # print('dQ', dQ.shape)
    # print('dQ0', dQ[0])
    # print('dQ1', dQ[-1])
    return list(Q), list(dQ) #result.position), list(result.velocity)

def joint_approach(q0, q1, speed=None):
    if speed is None:
        speed = cfg.joint_speed

    return joint_trajectory_time_optimal([q0, q1], speed) #, max_speed=geom.radians(speed))

def angdistance(a, b):
    return a.angdist(b, metric=3)*2



def linear_approach(x0, x1, duration=0,
                    angular_velocity=None,
                    cartesian_speed=None):
    ''' Approach trajectory in task space using a constant speed'''

    pose0 = to_pose(x0)
    pose1 = to_pose(x1)

    if angular_velocity is None:
        angular_velocity = cfg.angular_velocity
    if cartesian_speed is None:
        cartesian_speed = cfg.cartesian_speed

    if duration <= 0.0:
        duration = duration_from_distance(pose0, pose1,
                                          angular_velocity,
                                          cartesian_speed)

    # angular_approach_step = angular_velocity*cfg.dt
    # pos_approach_step = cartesian_speed*cfg.dt
    # if duration==0:
    #     angdist = abs(pose0.angdist(pose1))
    #     dist = np.linalg.norm(pose0.t - pose1.t)
    #     # Compute approach steps based on distance
    #     ang_approach_steps = int(angdist/angular_approach_step)
    #     pos_appro ach_steps = int(dist/pos_approach_step)
    #     n = max(ang_approach_steps, pos_approach_steps)
    # else:
    n = int(duration/cfg.dt)
    # Compute appriach keypoints if necessary
    if n > 0:
        approach = SE3.interp(pose0, pose1, np.linspace(0, 1, n))
        #approach = pose0.interp(pose1, n)
        #approach = [np.concatenate([pose.t, np.array(UnitQuaternion(pose))]) for pose in approach]
        return approach
    return []

def trapezoidal_approach(x0, x1,
                         angular_velocity=None,
                         cartesian_speed=None,
                         debug=False):
    ''' Approach trajectory in task space using a trapezoidal speed profile'''
    return plan_poses_time_optimal([x0, x1])

    pose0 = to_pose(x0)
    pose1 = to_pose(x1)
    print('Trapezoidal approach from', pose0, 'to', pose1)

    #print(np.array2string(np.array(pose0), separator=','))
    #print(np.array2string(np.array(pose1), separator=','))
    #print(repr(np.array(pose1)))
    #print('pose0 valid', SE3.isvalid(np.array(pose0)))
    #print('pose1 valid', SE3.isvalid(np.array(pose1)))

    angdist = abs(angdistance(pose0, pose1)) #pose0.angdist(pose1))
    dist = np.linalg.norm(pose0.t - pose1.t)

    if angular_velocity is None:
        angular_velocity = cfg.angular_velocity

    if cartesian_speed is None:
        cartesian_speed = cfg.cartesian_speed

    # Use angular velocity if it is the limiting factor
    if angdist/angular_velocity > dist/cartesian_speed:
        v_max = angular_velocity # * angdist
        d = angdist
    else:
        v_max = cartesian_speed
        d = dist

    # Use trapezoidal planner to compute trajectory parameterization
    # Might not be ideal for orientations?
    a_max = v_max/cfg.acceleration_time

    dt = cfg.dt
    tr = trapezoidal.trapezoidal_profile([0.0], [d], [0], [0], [v_max], [a_max])
    timesteps = int(tr.time / dt)
    time = np.linspace(0, tr.time, timesteps)
    ts = [tr(t)[0][2]/d for t in time] # Assumin

    #print('trapezoidal ts:', ts)
    #print(pose0, pose1)
    approach = SE3_interp(pose0, pose1, ts)
    if len(approach):
        pass #print('approach', to_pose(approach[0]), 'to ', to_pose(approach[-1] ))
    else:
        print('empty approach')
    if approach:
        res = approach
        #print('diffs', to_pos_orn(x0) - res[0], to_pos_orn(x1) - res[-1])
        #res = [np.array(keypoints[0])] + res + [np.array(keypoints[-1])]
    return approach

    # approach = [pose0.interp(pose1, t) for t in ts]
    # return pose_traj_to_posorn(approach)
def plan_poses_time_optimal(keypoints, max_speed=None, max_angular_velocity=None, accel_time=None, tol=0.0):
    #assert(False)
    if max_speed is None:
        max_speed = cfg.cartesian_speed
    if max_angular_velocity is None:
        max_angular_velocity = cfg.angular_velocity
    if accel_time is None:
        accel_time = cfg.acceleration_time

    from trajectory_smoothing import TrajectorySmoother

    # Convert to relative axis angle representation
    # interpolation will occur along specified angles
    keypoints = [to_pos_orn(to_pose(p)) for p in keypoints]
    #keypoints = [keypoints[0]] + keypoints + [keypoints[-1]]
    n = len(keypoints)

    angles = [0.0]
    axes = []
    orientations = []
    for a, b in zip(keypoints, keypoints[1:]):
        q1 = UnitQuaternion(a[3:])
        q2 = UnitQuaternion(b[3:])
        angle, axis = (q2 * q1.inv()).angvec()
        angles.append(angle)
        axes.append(axis)
        orientations.append(q1)
    orientations.append(UnitQuaternion(keypoints[-1][3:]))
    angles = np.cumsum(angles)

    keypoints_angle = [np.concatenate([p[:3], [a]]) for p, a in zip(keypoints, angles)]

    dof = 4
    max_velocity = np.concatenate([geom.normalize(np.ones(3, dtype=np.float32))*max_speed,
                                   [max_angular_velocity]])
    max_acceleration = max_velocity/accel_time

    smoother = TrajectorySmoother(
        dof, max_velocity, max_acceleration, tol
    )

    result = smoother.smooth_interpolate(np.array(keypoints_angle), interpolation_dt=cfg.dt, traj_dt=0.001)

    res = []
    for p, idx in zip(result.position, result.segment_indices):
        angle = p[3] - angles[idx]
        aa = UnitQuaternion.AngVec(angle, axes[idx])
        o = aa * orientations[idx]
        res.append(np.concatenate([p[:3], np.array(o)]))

    #print('trap vel', result.velocity)
    if res:
        pass
        #print('diffs', keypoints[0] - res[0], keypoints[-1] - res[-1])
        #res = [np.array(keypoints[0])] + res + [np.array(keypoints[-1])]
    return np.array(res)

def approach(a, b, *args, **kwargs):
    return trapezoidal_approach(a, b, *args, **kwargs)

def approach_sequence(seq, *args, **kwargs):
    if len(seq)==2:
        return approach(seq[0], seq[1], *args, **kwargs)
    return plan_poses_time_optimal(seq, *args, **kwargs)
    # #path, _, _ = plan_polylines_time_optimal([np.array(seq)], *args, **kwargs)
    # traj = []
    # for a, b in zip(seq, seq[1:]):
    #     traj += approach(a, b, *args, **kwargs)
    # return traj

def pose_traj_to_posorn(trajectory):
    ''' Convert a trajectory made of SE3 poses to pos+quaternions'''
    return [np.concatenate([pose.t, np.array(UnitQuaternion(pose))]) for pose in trajectory]

def cleanup(traj, tol=0.01):
    ''' Remove redundant waypoints'''
    if len(traj) < 2:
        return traj
    res = [traj[0]]
    for a, b in zip(traj, traj[1:]):
        if np.linalg.norm(res[-1] - b) > tol:
            res.append(b)
    return np.array(res)

glob = lambda: None

def plan_bezier_chains_time_optimal(chains, max_speed, accel_time, subd=5, angle_tolerance=5, tol=10/1000.0, bezier_tol=0):
    samples = []
    for chain in chains:
        for chunk in bezier.split_chain(chain, angle_tolerance):
            if bezier_tol > 0:
                P = bezier.cubic_bezier_adaptive(chain, bezier_tol)
            else:
                P = bezier.bezier_piecewise(chain, subd=subd)
            samples.append(P)
    return plan_polylines_time_optimal(samples, max_speed, accel_time, tol)


def plan_polylines_time_optimal(polylines, max_speed, accel_time, tol=10/1000.0):
    from trajectory_smoothing import TrajectorySmoother
    dof = len(polylines[0][0])

    max_accel = max_speed / accel_time
    # Max velocity and acceleration combined along unit vector
    # Note that otherwise we would get higher max speed and acceleration maginitude
    max_velocity = geom.normalize(np.ones(dof, dtype=np.float32)) * max_speed
    max_acceleration = geom.normalize(np.ones(dof, dtype=np.float32)) * max_accel
    #max_velocity = np.ones(dof, dtype=np.float32) * max_speed
    #max_acceleration = np.ones(dof, dtype=np.float32) * max_accel


    config.state.progress_max = len(polylines)
    config.state.progress = 0
    trajectories = []
    velocities = []
    accelerations = []
    for idx, P in enumerate(polylines):
        # Traj dt used for computations
        nattempt = 2
        print("%d of %d: Planning %d points"%(idx+1, len(polylines), len(P)))
        smoother = TrajectorySmoother(
        dof, max_velocity, max_acceleration, tol
        )

        for i in range(nattempt):
            P = np.array(P)
            result = smoother.smooth_interpolate(P, interpolation_dt=cfg.dt, traj_dt=0.001)
            if not result.success:
                print('Smoothing failed in trajectory smoothing, attempt %d of %d'%(i+1, nattempt))
                print(max_speed, accel_time, tol)
                Q = np.array(P) #[:,:2]
                glob.Q = Q
                glob.P = np.array(P)
                P = geom.uniform_sample(P, 1.0/1000)

                # A = geom.degrees(geom.turning_angles(Q))
                # L = geom.chord_lengths(Q)
                # print('Length', len(Q))
                # print('Min angle:', np.min(A))
                # print('Max angle:', np.max(A))
                # print('Min length:', np.max(L))
                # print('Max length:', np.max(L))
                # P = cleanup(P, tol)
                #P = np.array([P[j] for j in range(0, len(P), 2)])
                # smoother = TrajectorySmoother(
                #     dof, max_velocity, max_acceleration, tol
                # )

            else:
                print("Success %d"%(i+1))
                print("%d points"%len(result.position))
                trajectories.append(result.position)
                velocities.append(result.velocity)
                accelerations.append(result.acceleration)
                break
        config.state.progress = idx+1

    print("Finished planning")
    return trajectories, velocities, accelerations


def plan_bezier_chains_jerk(chains, max_speed, max_accel_time=0.5, angle_tolerance=10, subd=3, maxiter=300, tol=0.0, bezier_tol=0):
    '''Given a series (2d) of cubic Bezier chains, compute minimum jerk trajectories that track the curves
    with a given maximum speed and max acceleration time
    '''
    trajectories = []
    for chain in chains:
        chunk_traj = []
        for chunk in bezier.split_chain(chain, angle_tolerance):
            #if tol > 0:
            #    pts = bezier.bezier_piecewise(chunk, subd=20)
            #    beziers = fitCurve(pts, tol)
            traj, Q = plan_bezier_chain_jerk(chunk, max_speed, max_accel_time, subd, maxiter, bezier_tol=bezier_tol)
            chunk_traj.append(traj)
        trajectories.append(np.vstack(chunk_traj))
        #trajectories += chunk_traj
    return trajectories


def plan_polylines_jerk(polylines, simp, max_speed, max_accel_time=0.5, angle_tolerance=10, subd=3, maxiter=300, tol=0.0):
    '''Given a series (2d) of cubic Bezier chains, compute minimum jerk trajectories that track the curves
    with a given maximum speed and max acceleration time
    '''
    trajectories = []
    simplified = []
    config.state.progress = 0
    config.state.progress_max = len(polylines)
    for i, P in enumerate(polylines):
        config.state.progress = i+1
        traj, Q = plan_jerk(P, simp, max_speed, max_accel_time, subd, maxiter)
        trajectories.append(traj)
        simplified.append(Q)
    return trajectories, simplified


def plan_jerk(P, simp, max_speed, max_accel_time, subd=2, maxiter=300):
    # Compute points along curve
    old = P
    P = np.array(P)
    dt = cfg.dt
    dim = P.shape[1]
    print('Input polyline with %d points'%len(P))
    if simp > 0:
       _, I = simplify.dce(P[:,:2], simp, get_indices=True)
       P = P[I]
    #print('simp', P, simp)
    # Ps = P
    # #Ps = simplify.dce(P, simplify_tol)
    # #Ps = simplify.dp_simplify(P, simplify_tol*10)

    # if len(Ps) > 2:
    #     P = Ps
    # else:
    #     P = np.vstack([P[0], (P[0] + P[-1])/2, P[-1]])
    # Path-constrained minimum jerk according to Todorov and Jordan
    # we use this to get passage times, velocity and acceleration that minimize jerk
    print('Planning minimum jerk trajectory for %d via-points (simplified)'%len(P))
    n = max(len(P)*2, 100) #chain)* #10 # Pretty arbitrary
    traj, psg, Q, dQ, ddQ = todorov.min_jerk_todorov(P, n, get_vel_acc=True, maxiter=maxiter)
    opsg = psg

    # Check that passage times are strictly increasing
    if len(psg) > 1:
        dpsg = np.maximum(np.diff(psg), 1e-3)
        if np.any(dpsg < 0):
            print("Negative delta-time")
        psg = np.cumsum(np.concatenate([[psg[0]], dpsg]))
        if psg[-1] > n:
            print("Passage times exceed total time")
            psg = psg/psg[-1]*(n-1e-3)

    # Add zero boundary conditions and initial and final time
    dQ = np.vstack([np.zeros(dim), dQ, np.zeros(dim)])
    ddQ = np.vstack([np.zeros(dim), ddQ, np.zeros(dim)])
    T = np.concatenate([[0], psg, [n]])

    # Use passage times, velocity and acceleration to compute piecewise polynomial
    derivs = np.dstack([Q, dQ, ddQ])
    bps = []
    try:
        for i in range(dim):
            bps.append(BPoly.from_derivatives(T, np.vstack([derivs[:, i, j] for j in range(3)]).T))
    except ValueError as e:
        print(e)
        print('T', T)
        pdb.set_trace()

    dbps = [bp.derivative() for bp in bps]
    ddbps = [dbp.derivative() for dbp in dbps]

    # Estimate maximum speed and acceleration
    x_test = np.linspace(0, T[-1], n)
    Y = np.vstack([bp(x_test) for bp in bps]).T
    dY = np.vstack([bp(x_test) for bp in dbps]).T
    ddY = np.vstack([bp(x_test) for bp in ddbps]).T

    smax = np.max(rownorm(dY))
    amax = np.max(rownorm(ddY))

    # Compute time step that satisfies max speed and acceleration bounds
    max_accel = max_speed/max_accel_time
    div_s = max_speed/smax
    div_a = np.sqrt(max_accel/amax)
    div = min(div_s, div_a)
    #print('smax, amax', smax, amax)
    #print('div', div_s, div_a)

    x = np.arange(0, T[-1], dt*div)
    # Compute trajectory
    Y = np.vstack([bp(x) for bp in bps]).T
    if len(Y)==2:
        pdb.set_trace()
    return Y, P


def plan_bezier_chain_jerk(chain, max_speed, max_accel_time, subd=2, maxiter=300, bezier_tol=0):
    '''Given a (2d) cubic Bezier chain, compute a minimum jerk trajectory that tracks the curve
    with a given maximum speed and max acceleration time
    '''
    # Compute points along curve
    dt = cfg.dt
    chain = np.array(chain)
    dim = chain.shape[1]
    if bezier_tol > 0:
        P = bezier.cubic_bezier_adaptive(chain, bezier_tol)
    else:
        P = bezier.bezier_piecewise(chain, subd=subd)
    return plan_jerk(P, 0.0, max_speed, max_accel_time, subd, maxiter)

def bezier_to_bpolys(chain):
    X = []
    dX = []
    chain = np.array(chain)
    dim = len(chain[0])
    n = len(chain)
    #chain = list(chain)
    #chain[1] = chain[0]
    #chain[-2] = chain[-1]
    #chain = [chain[0]]*3 + chain + [chain[-1]*3]
    beziers = bezier.chain_to_beziers(chain) #[:5]
    beziers[0][1] = np.array(beziers[0][0])
    beziers[-1][-2] = np.array(beziers[-1][-1])

    #b0 = [np.array(beziers[0][0])]*4
    #b1 = [np.array(beziers[-1][-1])]*4
    #beziers = [b0] + beziers + [b1]
    #print(beziers[:3])
    amt = 3
    for bez in beziers:
        X.append(bez[0])
        dX.append((bez[1] - bez[0])*amt)
    X.append(bez[-1])
    dX.append((bez[-1] - bez[-2])*amt)

    # for i in range(0, n, 3):
    #     X.append(chain[i])
    #     if i < n-1:
    #         dX.append((chain[i+1] - chain[i])*3)
    #     else:
    #         dX.append((chain[i] - chain[i-1])*3)
    X = np.array(X)
    dX = np.array(dX)
    #print(dX.shape)
    T = np.linspace(0, len(X)-1, len(X))
    bps = []
    for i in range(dim):
        bps.append(BPoly.from_derivatives(T, np.vstack([X[:,i], dX[:, i]]).T))

    bps2 = []

    for i in range(dim):
        bps2.append(BPoly.from_derivatives(T, np.vstack([bps[i](T)] + [bps[i].derivative(k+1)(T) for k in range(1)]).T))

    return T[0], T[-1], bps

def resample_bezier(Cp, max_speed, accel_time, get_t=False):
    dt = cfg.dt
    t0, t1, bps = bezier_to_bpolys(Cp)

    dbps = [bp.derivative() for bp in bps]
    ddbps = [dbp.derivative() for dbp in dbps]

    # Estimate maximum speed and acceleration
    x_test = np.linspace(t0, t1, len(Cp)*10) #0, T[-1], n)
    Y = np.vstack([bp(x_test) for bp in bps]).T
    dY = np.vstack([bp(x_test) for bp in dbps]).T
    ddY = np.vstack([bp(x_test) for bp in ddbps]).T

    smax = np.max(rownorm(dY))
    amax = np.max(rownorm(ddY))

    # Compute time step that satisfies max speed and acceleration bounds
    max_accel = max_speed/accel_time
    #print('max speed', max_speed)
    #print('max accel', max_accel)
    div_s = max_speed/smax
    div_a = np.sqrt(max_accel/amax)
    div = min(div_s, div_a)

    x = np.arange(t0, t1, dt*div)
    # Compute trajectory
    Y = np.vstack([bp(x) for bp in bps]).T
    if get_t:
        return x, Y
    return Y

def rownorm(dP):
    return np.sqrt(np.sum(dP**2, axis=1)) 


def compute_speed(X, dt):
    D = np.diff(X, axis=0)/dt
    return rownorm(D)


def compute_accel(X, dt):
    D = np.diff(X, axis=0, n=2)/(dt**2)
    return rownorm(D)


def lifting_motion(a, b, height, max_speed=150/1000, max_accel_time=0.4):
    ''' Create a pen lifting motion between two points assumed to lie on the XY plane'''
    dt = cfg.dt
    a = np.array(a)
    b = np.array(b)
    d = b-a
    up = [0, 0, height]
    if geom.distance(a, b) < height:
        P = np.array([a,  a+d*(1/3)+up, b-d*(1/3)+up, b])
    else:
        dd = geom.normalize(d)
        P = np.array([a, a+dd*height*0.5 + up, (a+b)/2 + up, b-dd*height*0.5+up, b])
    
    paths, _, _ = plan_polylines_time_optimal([P],
                                              max_speed=max_speed,
                                              accel_time=max_accel_time,
                                              tol=0.3)
    return paths[0]

    #u = geom.cum_chord_lengths(P)
    u = np.linspace(0, 1.0, len(P)) 
    #u = u/u[-1]
    bc_type='clamped'
    spl = CubicSpline(u, P) #, bc_type=bc_type)
    dspl = spl.derivative()
    ddspl = spl.derivative(2)
    x_test = np.linspace(0, 1, 100)
    Y = spl(x_test)
    dY = dspl(x_test)
    ddY = ddspl(x_test)
    smax = np.max(rownorm(dY))
    amax = np.max(rownorm(ddY))
    print('amax', amax)
    max_accel = max_speed/max_accel_time
    div_s = max_speed/smax
    div_a = np.sqrt(max_accel/amax)
    div = min(div_s, div_a)
    t = np.arange(0, 1, dt*div)
    if len(t) <= 2:
        pdb.set_trace()
    X = spl(t)
    speed = compute_speed(X, dt)
    accel = compute_accel(X, dt)
    dY = spl.derivative()(x_test)
    ddY = spl.derivative(2)(x_test)
    A = np.diff(X, n=2, axis=0)/(dt**2)
    print("max accel", np.max(accel), 'a', np.abs(np.max(A)), 'da', np.abs(np.max(ddY)), 'vs', max_accel)
    if np.max(speed) > max_speed:
        print("Exceeding max speed", np.max(speed), max_speed)
        #pdb.set_trace()
    if np.max(accel) > max_accel:
        print("Exceeding max accel", np.max(accel), max_accel)
        #pdb.set_trace()
    return spl(t)


def tsp_glyph(n, r1=0.5, r2=1.0, center=[0,0]):
    P = geom.shapes.random_radial_polygon(n, r1, r2, center=center)
    I = inverse_path_tsp(P)
    return P[I,:]


def plan_polylines_lqt(S, max_speed, accel_time, penups=True, penup_variability=0.4, **kwargs):
    trajectories = []
    for P in S:
        variability = []
        # if penups:
        #     variability = [1, penup_variability] + [1 for _ in P[1:-1]] + [penup_variability, 1]

        trajectory = plan_lqt(P, max_speed, accel_time, variability=variability, **kwargs)
        trajectories.append(trajectory)
    return trajectories

def plan_lqt(P, max_speed, accel_time, **kwargs):
    ''' Do semitied LQT planning with max speed and accel constraints'''
    dim  = P.shape[1]
    default_params = dict(angle=0,
                          sigma=1.0,
                          order=3,
                          opt_x0=False,
                          stop_motion=True,
                          isotropy=1,
                          activation_sigma=0.5,
                          subd=5) # Assuming duration of 0.5
    default_params.update(kwargs)
    X, duration = semitied_lqt(P, dim=dim, **default_params)

    T = np.linspace(0, duration, X.shape[0])
    nderiv = X.shape[1]//dim

    # Construct piecewise polynomial for each dimension
    bp = [BPoly.from_derivatives(T,
                np.vstack([X[:, j*dim+i] for j in range(nderiv)]).T)
          for i in range(dim)]
    # And velocity/accel
    dbp = [p.derivative(1) for p in bp]
    ddbp = [p.derivative(2) for p in bp]

    # Estimate maximum speed and acceleration norm for current parameterization
    max_accel = max_speed / accel_time

    dX = np.vstack([p(T) for p in dbp]).T
    ddX = np.vstack([p(T) for p in ddbp]).T
    smax = np.max(rownorm(dX))
    amax = np.max(rownorm(ddX))

    # Get ratio to reach desired max velocity or acceleration
    div_s = max_speed/smax
    div_a = np.sqrt(max_accel/amax)
    div = min(div_s, div_a)

    # Get new parametrization to achieve desired max, given the timestep
    dt = cfg.dt
    x = np.arange(0.0, T[-1], dt*div)

    Y = np.vstack([p(x) for p in bp]).T
    return Y

def plan_lqt_bezier(Cp, ctrl_ratio, max_speed, accel_time, **kwargs):
    ''' Do semitied LQT planning with max speed and accel constraints'''
    dim  = Cp.shape[1]
    default_params = dict(order=3,
                          opt_x0=False,
                          stop_movement=True,
                          activation_sigma=0.5,
                          subd=5) # Assuming duration of 0.5
    default_params.update(kwargs)
    X, duration = bezier_lqt(Cp, ctrl_ratio, **default_params)
    T = np.linspace(0, duration, X.shape[0])
    nderiv = X.shape[1]//dim

    # Construct piecewise polynomial for each dimension
    bp = [BPoly.from_derivatives(T,
                np.vstack([X[:, j*dim+i] for j in range(nderiv)]).T)
          for i in range(dim)]
    # And velocity/accel
    dbp = [p.derivative(1) for p in bp]
    ddbp = [p.derivative(2) for p in bp]

    # Estimate maximum speed and acceleration norm for current parameterization
    max_accel = max_speed / accel_time

    dX = np.vstack([p(T) for p in dbp]).T
    ddX = np.vstack([p(T) for p in ddbp]).T
    smax = np.max(rownorm(dX))
    amax = np.max(rownorm(ddX))

    # Get ratio to reach desired max velocity or acceleration
    div_s = max_speed/smax
    div_a = np.sqrt(max_accel/amax)
    div = min(div_s, div_a)

    # Get new parametrization to achieve desired max, given the timestep
    dt = cfg.dt
    x = np.arange(0.0, T[-1], dt*div)

    Y = np.vstack([p(x) for p in bp]).T
    return Y

def bezier_lqt(Cp, ctrl_ratio=7.5, **kwargs):
    ''' Mimic a cubic Bezier curve with isotropic Gaussians and a stepwise tracking reference'''
    params = dict(activation_sigma=0.5,
                  d=3.,
                  order=3,
                  sigma=10,
                  segment_duration=0.5,
                  opt_x0=False,
                  stop_movement=True,
                  periodic=False,
                  iterative=True,
                  subd=10,
                  alpha=0
                )
    params.update(**kwargs)
    d = params['d']
    sigma = params['sigma']

    Cp = np.array(Cp).T
    glob_scale = 1.0
    m = Cp.shape[1]
    dim = Cp.shape[0]
    Cp = Cp*1
    d = d*glob_scale
    sigma = sigma*glob_scale
    Mu = np.zeros((dim,m))
    Sigma = np.zeros((dim,dim,m))
    ctrl_ratio *= ctrl_ratio
    sigma = sigma**2
    num_bezier = int((m - 1) / 3)
    for i in range(num_bezier):

        p1,p2,p3,p4 = [Cp[:,j+i*3] for j in range(4)]

        dv = (p1 - p2 + p4 - p3)/2

        dr = 0.0
        Mu[:,i*3] = p1
        Mu[:,i*3+3] = p4
        Mu[:,i*3+1] = p2 + dv*dr #- d1*dr
        Mu[:,i*3+2] =  p3 + dv*dr #- d2*dr

        Sigma[:,:,i*3]   = np.eye(dim)*sigma
        Sigma[:,:,i*3+1] = np.eye(dim)*sigma*ctrl_ratio
        Sigma[:,:,i*3+2] = np.eye(dim)*sigma*ctrl_ratio
        Sigma[:,:,i*3+3] = np.eye(dim)*sigma

    #Sigma[:,:,0] = np.eye(dim)*2 #0.05
    #Sigma[:,:,-1] = np.eye(dim)*2 #0.05

    iterative = params['iterative']
    if params['periodic']:
        iterative = False
    sys = lqt.DynamicalSystem(params['order'], params['subd'], segment_duration=params['segment_duration'], dim=dim)
    print("Stop?", params['stop_movement'])
    ref = lqt.cartesian_reference(sys, Mu, Sigma,
                                  activation_sigma=params['activation_sigma'],
                                  stop_movement=params['stop_movement'],
                                  clamp=1,
                                  alpha=params['alpha'],
                                  iterative=iterative)
    if iterative:
        res = lqt.solve_lqt_iterative(sys, ref, params['d'])
    else:
        res = lqt.solve_lqt(sys, ref, params['d'])

    X = res['x'].T
    duration = (X.shape[0]-1)*sys.dt
    return X, duration


def semitied_lqt(P, **kwargs):
    params = dict(angle=0,
                  isotropy=0.5,
                  sigma=1,
                  activation_sigma=0.5,
                  d=3.,
                  order=4,
                  segment_duration=0.5,
                  opt_x0=False,
                  stop_movement=True,
                  periodic=False ,
                  iterative=False,
                  subd=10,
                  clamp=0,
                  dim=2,
                  variability=[],
                )
    params.update(**kwargs)
    P = P.T
    dim = params['dim']
    Sigma = []
    cov = np.eye(dim)*((0.9)**2) # Penup hack
    cov[:2,:2] = lqt.make_sigma(geom.radians(params['angle']), [params['isotropy']*params['sigma'], params['sigma']])

    for p in P.T:
        Sigma.append(cov)
    # if params['variability']:
    #     print(params['variability'])
    #     Sigma = [w*s for w, s in zip(params['variability'], Sigma)]

    iterative = params['iterative']
    if params['periodic']:
        iterative = False

    Sigma = np.array(Sigma).T
    sys = lqt.DynamicalSystem(params['order'], params['subd'], segment_duration=params['segment_duration'], dim=params['dim'])
    ref = lqt.cartesian_reference(sys, P, Sigma,
                                    activation_sigma=params['activation_sigma'],
                                    stop_movement=params['stop_movement'], periodic=params['periodic'],
                                    iterative=iterative,
                                    clamp=params['clamp'])
    if iterative:
        res = lqt.solve_lqt_iterative(sys, ref, params['d'])
    else:
        res = lqt.solve_lqt(sys, ref, params['d'])

    # res = lqt.solve_lqt(sys, ref, params['d'], opt_x0=params['opt_x0']) #, r=1e-43)
    X = res['x'].T
    duration = (X.shape[0]-1)*sys.dt
    return X, duration

def distance(a, b):
    return np.linalg.norm(b[:2] - a[:2])


def path_tsp(P, distfn=distance, **kwargs):
    from networkx.algorithms.approximation import traveling_salesman_problem
    G = nx.Graph()
    n = len(P)
    for i in range(n):
        G.add_node(i)
    # add weighted edges
    for a, b in combinations(range(n), 2):
        G.add_edge(a, b, weight=distfn(P[a], P[b])) #1.0/(1+distance(P[a], P[b])))
    I = traveling_salesman_problem(G, cycle=False, **kwargs)
    return I


def inverse_path_tsp(P, **kwargs):
    def invdist(a, b):
        return 1.0/(1+distance(a, b))
    return path_tsp(P, invdist, **kwargs)


def gauss_easing(t, easing=2):
    x = erf((t-0.5)*easing)
    x = (x-np.min(x)) / (np.max(x)-np.min(x))
    return x


def gaussspace(a, b, n, easing=2):
    t = np.linspace(a, b, n-2)
    # Hack for zero velocity at start and end
    t = np.concatenate([[a], t, [b]])
    return gauss_easing(t, easing)


def expspace(a, b, n, easing=2):
    t = np.linspace(0, 1, n)
    t = (t-0.5)
    v = np.exp(-(t**2)*(easing**2)) #/(easing**2))
    v = v/np.sum(v)
    return v


def gauss_easeinout(a, b, duration, ease_duration=0.25, easing=5):
    ''' Returns a gaussian easing function from a to b with given duration and easing_duration'''
    ease_duration = min(duration/2, ease_duration)
    n = int(duration/cfg.dt)
    nease = int(ease_duration/cfg.dt)
    n = max(n, nease*2)
    gauss = expspace(0, 1, nease*2, easing=easing)
    gauss = gauss/np.max(gauss)
    t = [0.0] + list(gauss[:nease]) + list(np.ones(n-nease*2)) + list(gauss[nease:])
    t = np.cumsum(t)
    t = t/np.max(t)
    return a + (b - a)*t

def bspline2(n, pts, closed=False, alpha=1, degree=3, smooth_k=0, w=None, der=0, duration=1.0, clamped=True):
    pts = np.array(pts)
    if closed:
        pts = np.vstack([pts, pts[0]])
    if w is None:
        w = np.ones(len(pts))
    elif is_number(w):
        w = np.ones(len(pts))*w
        if closed:
            w = np.concatenate([w, [w[0]]])

    dim = pts.shape[1]
    u = np.linspace(0, duration, pts.shape[0])
    # print('dt_pre:', u[1] - u[0])
    degree = min(degree, pts.shape[0]-1)
    # if clamped:
    #     bc_type=([(1, np.zeros(dim))], [(1, np.zeros(dim))])
    # else:
    #     # Natural
    #     bc_type=([(2, 0.0)], [(2, 0.0)])
    #spl = make_interp_spline(u, pts, k=degree) #, bc_type=bc_type)
    tck, _ = splprep(pts.T, u=u, k=degree, per=closed, s=smooth_k, w=w)

    t = np.linspace(0, duration, n)
    res = [np.vstack(splev(t, tck)).T]

    #print(t[1] - t[0])
    #print('dt_post:', t[1] - t[0])
    for i in range(der):
        res += [np.vstack(splev(t, tck, der=i+1)).T]
    return res

def bspline(n, pts, closed=False, alpha=1, degree=3, smooth_k=0, w=None, der=0, duration=1.0, clamped=True):
    pts = np.array(pts)
    if closed:
        pts = np.vstack([pts, pts[0]])
    if w is None:
        w = np.ones(len(pts))
    elif is_number(w):
        w = np.ones(len(pts))*w
        if closed:
            w = np.concatenate([w, [w[0]]])

    dim = pts.shape[1]
    u = np.linspace(0, duration, pts.shape[0])
    print('dt_pre:', u[1] - u[0])
    degree = min(degree, pts.shape[0]-1)
    if clamped:
        bc_type=([(1, np.zeros(dim))], [(1, np.zeros(dim))])
    else:
        pass
        # Natural
    #bc_type=([(2, 0.0)], [(2, 0.0)])
    spl = make_interp_spline(u, pts, k=degree) #, bc_type=bc_type)
    #tck, _ = splprep(pts.T, u=u, k=degree, per=closed, s=smooth_k, w=w)

    t = np.linspace(0, duration, n)
    res = [spl(t)]

    #print(t[1] - t[0])
    #print('dt_post:', t[1] - t[0])
    for i in range(der):
        spl = spl.derivative()
        res.append(spl(t))
    return res

    # if type(der) == list:
    #     res = []
    #     for d in der:
    #         res.append(np.vstack(splev(t, tck, der=d)).T)
    #     return res

    # res = splev(t, tck, der=der)
    # return np.vstack(res).T


def saturate_speed(trajectory, max_speed, dt):
    Xsat = []
    for a, b in zip(trajectory, trajectory[1:]):
        d = b - a
        speed = np.linalg.norm(d) / dt
        if speed > max_speed:
            nk = int(np.round(speed/max_speed))
            Xsat += [a + d*t for t in np.linspace(0, 1, nk+1)[:-1]]
        else:
            Xsat.append(a)

    Xsat.append(trajectory[-1])
    return np.array(Xsat)


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

def align_pos_z(pose, z):
    pose = to_pose(pose)
    pose = align_frame_z(pose, z)
    return to_pos_orn(pose)

def align_frame_z(frame, z):
    z = np.array(z)
    frame = np.array(frame)
    frame[:3,2] = z
    frame[:3,0] = np.cross(frame[:3,1], z)
    frame[:3,1] = np.cross(z, frame[:3,0])
    # Must make sure we don't get precision issues
    # https://github.com/bdaiinstitute/spatialmath-python/issues/28
    return trnorm(frame)

def align_pos_zy(pose, z, y):
    pose = to_pose(pose)
    pose = align_frame_zy(pose, z, y)
    return to_pos_orn(pose)

def align_pose_zy(pose, z, y):
    pose = to_pose(pose)
    pose = align_frame_zy(pose, z, y)
    return to_pose(pose)

def align_frame_zy(frame, z, y):
    z = np.array(z)
    y = np.array(y)
    frame = np.array(frame)
    frame[:3,2] = z
    frame[:3,0] = np.cross(y, z)
    frame[:3,1] = np.cross(z, frame[:3,0])
    # Must make sure we don't get precision issues
    # https://github.com/bdaiinstitute/spatialmath-python/issues/28
    return trnorm(frame)
