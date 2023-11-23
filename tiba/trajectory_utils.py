#!/usr/bin/env python3
from importlib import reload
import numpy as np
import networkx as nx
from polygonsoup import geom, bezier, simplify
from itertools import combinations
from scipy.special import erf
from spatialmath import SE3, SO3, UnitQuaternion
from spatialmath.base import trnorm
from scipy.interpolate import interp1d, splprep, splev, make_interp_spline, CubicSpline, BPoly
from roboticstoolbox.tools.trajectory import ctraj, trapezoidal_func, trapezoidal
from . import trapezoidal, todorov
reload(trapezoidal)
from .fitCurves.fitCurves import fitCurve

from .config import cfg
import numbers
import pdb

def is_number(x):
    return isinstance(x, numbers.Number)

def to_pos_orn(pose):
    if (type(pose) == list or
        (type(pose) == np.ndarray and len(pose.shape) == 1)):
        return pose
        #return pose.t, np.array(UnitQuaternion(pose))
    pose = SE3(pose)
    pose = pose.norm()
    return np.concatenate([pose.t, np.array(UnitQuaternion(pose))])

def to_pose(pose):
    # TODO Add normalization?
    if type(pose) == SE3:
        return pose.norm()
    if type(pose) == np.ndarray and len(pose.shape) == 2:
        return SE3(pose).norm()
    pose = SE3(*pose[:3])*(UnitQuaternion(pose[3:])).SE3()
    return pose.norm()


def append_rest(traj, duration):
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
    traj = list(traj)
    n = int(duration/cfg.dt)
    if not n:
        return traj
    return [traj[0]]*n + traj + [traj[-1]]*n

# Trajectory generators
def xy_circles_mm(pose, radius_mm, cycle_duration, cycles=1, ease_duration=0.5, easing=2):
    pose = to_pos_orn(pose)
    pos, orn = pose[:3], pose[3:]

    duration = cycle_duration*cycles
    radius = radius_mm / 1000
    T = gauss_easeinout(0, np.pi*2*cycles, duration, ease_duration, easing=easing)
    pts = [pos + [np.cos(t)*radius, np.sin(t)*radius, 0] for t in T]
    return approach_inout(pose, [np.concatenate([pos, orn]) for pos in pts])

def offset_frame(frame, d):
    print(frame)
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


def joint_approach(q0, q1, speed=None):
    if speed is None:
        speed = cfg.default_joint_approach_speed

    q0 = np.array(q0)
    q1 = np.array(q1)

    dof = len(q0)
    dq = np.abs(q1-q0)
    d = np.max(dq)
    v_max = geom.radians(speed) / d
    a_max = v_max/cfg.acceleration_time
    dt = cfg.robot_dt
    tr = trapezoidal.trapezoidal_profile([0], [1],
                                         [0], [0],
                                         [v_max],
                                         [a_max])


    timesteps = int(tr.time / dt)
    print('timesteps', timesteps)
    print('time', tr.time)
    time = np.linspace(0, tr.time, timesteps)
    ts = [tr(t)[0][2] for t in time]
    dts = [tr(t)[0][1] for t in time]
    Qs = [q0 + (q1-q0)*t for t in ts]
    dQs = [dq*dt for dt in dts]
    print(dQs[0], dQs[-1])
    return Qs, dQs


def duration_from_distance(x0, x1, angular_approach_speed=None,
                                   cartesian_approach_speed=None):
    pose0 = to_pose(x0)
    pose1 = to_pose(x1)

    if angular_approach_speed is None:
        angular_approach_speed = cfg.default_angular_approach_speed
    if cartesian_approach_speed is None:
        cartesian_approach_speed = cfg.default_cartesian_approach_speed

    angdist = abs(pose0.angdist(pose1))
    dist = np.linalg.norm(pose0.t - pose1.t)

    def fitts_id(d, w=1):
        return np.log2(d/w + 1)

    duration = max(angdist/angular_approach_speed, dist/cartesian_approach_speed)
    print('fitts comp', duration, duration*fitts_id(duration))
    return duration * (1+fitts_id(duration)) # max(cfg.min_approach_duration, duration+fitts_id(duration))


def linear_approach(x0, x1, duration=0,
                    angular_approach_speed=None,
                    cartesian_approach_speed=None):
    pose0 = to_pose(x0)
    pose1 = to_pose(x1)

    if angular_approach_speed is None:
        angular_approach_speed = cfg.default_angular_approach_speed
    if cartesian_approach_speed is None:
        cartesian_approach_speed = cfg.default_cartesian_approach_speed

    if duration <= 0.0:
        duration = duration_from_distance(pose0, pose1,
                                          angular_approach_speed,
                                          cartesian_approach_speed)

    # angular_approach_step = angular_approach_speed*cfg.dt
    # pos_approach_step = cartesian_approach_speed*cfg.dt
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
        approach = pose0.interp(pose1, n)
        approach = [np.concatenate([pose.t, np.array(UnitQuaternion(pose))]) for pose in approach]
        return approach
    return []

def trapezoidal_approach(x0, x1,
                         angular_approach_speed=None,
                         cartesian_approach_speed=None,
                         debug=False):


    pose0 = to_pose(x0)
    pose1 = to_pose(x1)

    angdist = abs(pose0.angdist(pose1))
    dist = np.linalg.norm(pose0.t - pose1.t)

    if angular_approach_speed is None:
        angular_approach_speed = cfg.default_angular_approach_speed
    if cartesian_approach_speed is None:
        cartesian_approach_speed = cfg.default_cartesian_approach_speed

    if angdist/angular_approach_speed > dist/cartesian_approach_speed:
        v_max = angular_approach_speed # * angdist
        d = angdist
    else:
        v_max = cartesian_approach_speed
        d = dist

    a_max = v_max/cfg.acceleration_time

    dt = cfg.dt
    tr = trapezoidal.trapezoidal_profile([0.0], [d], [0], [0], [v_max], [a_max])
    timesteps = int(tr.time / dt)
    time = np.linspace(0, tr.time, timesteps)
    ts = [tr(t)[0][2]/d for t in time]
    approach = [pose0.interp(pose1, t) for t in ts]
    print('approach with ', len(approach), ' steps')
    return pose_traj_to_posorn(approach)

def approach(a, b, *args, **kwargs):
    return trapezoidal_approach(a, b, *args, **kwargs)

def pose_traj_to_posorn(trajectory):
    return [np.concatenate([pose.t, np.array(UnitQuaternion(pose))]) for pose in trajectory]

def gauss_approach(x0, x1, duration=-1,
                   angular_approach_speed=None,
                   cartesian_approach_speed=None,
                   easing=5,
                   debug=False):
    pose0 = to_pose(x0)
    pose1 = to_pose(x1)

    # if angular_approach_speed is None:
    #     angular_approach_speed = cfg.default_angular_approach_speed
    # if cartesian_approach_speed is None:
    #     cartesian_approach_speed = cfg.default_cartesian_approach_speed

    # if angular_approach_speed is None:
    #     angular_approach_speed = cfg.default_angular_approach_speed
    # if cartesian_approach_speed is None:
    #     cartesian_approach_speed = cfg.default_cartesian_approach_speed

    if duration <= 0.0:
        duration = duration_from_distance(pose0, pose1,
                                          angular_approach_speed,
                                          cartesian_approach_speed)

    # angular_approach_step = angular_approach_speed*cfg.dt
    # pos_approach_step = cartesian_approach_speed*cfg.dt

    # if duration<=0:
    #     print(pose0, pose1)
    #     dist = np.linalg.norm(pose0.t - pose1.t)
    #     print(dist)
    #     try:
    #         angdist = abs(pose0.angdist(pose1))
    #     except ValueError as e:
    #         print(e)
    #         angdist = 0

    #     if debug:
    #         pdb.set_trace()
    #     # Compute approach steps based on distance
    #     ang_approach_steps = int(angdist/angular_approach_step)
    #     pos_approach_steps = int(dist/pos_approach_step)
    #     n = max(ang_approach_steps, pos_approach_steps)
    # else:

    n = int(duration/cfg.dt)
    if n > 0:
        approach = [pose0.interp(pose1, t) for t in gaussspace(0, 1, n, easing=easing)]
        return pose_traj_to_posorn(approach)
    return []


def split_chain(chain, tol=10): #1e-1):
    tol = geom.radians(tol)
    #return [chain]
    beziers = bezier.chain_to_beziers(chain)
    chunks = [[beziers[0]]]
    bez = beziers[0]
    prev = bez[-1] - bez[-2]
    for bez in beziers[1:]:
        next = bez[1] - bez[0]
        theta = abs(geom.angle_between(prev, next))
        #print(theta)
        if theta > tol:
            #if np.linalg.norm(bez[0] - bez[1]) < tol:
            chunks.append([bez])
        else:
            chunks[-1].append(bez)
        prev = bez[-1] - bez[-2]
    return [bezier.beziers_to_chain(chunk) for chunk in chunks]

def plan_bezier_chains_jerk(chains, max_speed, max_accel_time=0.5, angle_tolerance=10, subd=3, maxiter=300, tol=0.0):
    '''Given a series (2d) of cubic Bezier chains, compute minimum jerk trajectories that track the curves
    with a given maximum speed and max acceleration time
    '''
    trajectories = []
    for chain in chains:
        chunk_traj = []
        for chunk in split_chain(chain, angle_tolerance):
            #if tol > 0:
            #    pts = bezier.bezier_piecewise(chunk, subd=20)
            #    beziers = fitCurve(pts, tol)
            traj = plan_bezier_chain_jerk(chunk, max_speed, max_accel_time, subd, maxiter)
            chunk_traj.append(traj)
        trajectories.append(np.vstack(chunk_traj))
        #trajectories += chunk_traj
    return trajectories


def plan_bezier_chain_jerk(chain, max_speed, max_accel_time, subd=5, maxiter=300):
    '''Given a (2d) cubic Bezier chain, compute a minimum jerk trajectory that tracks the curve
    with a given maximum speed and max acceleration time
    '''
    # Compute points along curve
    dt = cfg.dt
    chain = np.array(chain)
    dim = chain.shape[1]
    P = bezier.bezier_piecewise(chain, subd=subd)
    #P = simplify.dce(P, 0.001)
    # Ps = P
    # #Ps = simplify.dce(P, simplify_tol)
    # #Ps = simplify.dp_simplify(P, simplify_tol*10)

    # if len(Ps) > 2:
    #     P = Ps
    # else:
    #     P = np.vstack([P[0], (P[0] + P[-1])/2, P[-1]])
    # Path-constrained minimum jerk according to Todorov and Jordan
    # we use this to get passage times, velocity and acceleration that minimize jerk
    n = len(chain)*10 # Pretty arbitrary
    Q, psg, dQ, ddQ = todorov.min_jerk_todorov(P, n, get_vel_acc=True, maxiter=maxiter)
    opsg = psg
    # Check that passage times are strictly increasing
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
    derivs = np.dstack([P, dQ, ddQ])
    try:
        bpx = BPoly.from_derivatives(T, np.vstack([derivs[:,0, i] for i in range(3)]).T)
        bpy = BPoly.from_derivatives(T, np.vstack([derivs[:,1, i] for i in range(3)]).T)
    except ValueError as e:
        print(e)
        print('T', T)
        pdb.set_trace()

    dbpx = bpx.derivative()
    dbpy = bpy.derivative()
    ddbpx = dbpx.derivative()
    ddbpy = dbpy.derivative()

    # Estimate maximum speed and acceleration
    x_test = np.linspace(0, T[-1], n)
    Y = np.vstack([bpx(x_test), bpy(x_test)]).T
    dY = np.vstack([dbpx(x_test), dbpy(x_test)]).T
    ddY = np.vstack([ddbpx(x_test), ddbpy(x_test)]).T
    smax = np.max(colnorm(dY))
    amax = np.max(colnorm(ddY))

    # Compute time step that satisfies max speed and acceleration bounds
    max_accel = max_speed/max_accel_time
    div_s = max_speed/smax
    div_a = np.sqrt(max_accel/amax)
    div = min(div_s, div_a)
    #print('smax, amax', smax, amax)
    #print('div', div_s, div_a)

    x = np.arange(0, T[-1], dt*div)
    # Compute trajectory
    Y = np.vstack([bpx(x), bpy(x)]).T
    if len(Y)==2:
        pdb.set_trace()
    return Y

def colnorm(dP):
    return np.sqrt(dP[:,0]**2 + dP[:,1]**2)

def lifting_motion(a, b, height, max_speed=150/1000, max_accel_time=0.4):
    ''' Create a pen lifting motion between two points assumed to lie on the XY plane'''
    dt = cfg.dt
    a = np.array(a)
    b = np.array(b)
    d = b-a
    up = [0, 0, height]
    P = np.array([a,  a+d*(1/3)+up, b-d*(1/3)+up, b])
    u = geom.cum_chord_lengths(P)
    u = u/u[-1]
    bc_type='clamped'
    spl = CubicSpline(u, P, bc_type=bc_type)
    dspl = spl.derivative()
    ddspl = dspl.derivative()
    x_test = np.linspace(0, 1, 100)
    Y = spl(x_test)
    dY = dspl(x_test)
    ddY = ddspl(x_test)
    smax = np.max(colnorm(dY))
    amax = np.max(colnorm(ddY))
    max_accel = max_speed/max_accel_time
    div_s = max_speed/smax
    div_a = np.sqrt(max_accel/amax)
    div = min(div_s, div_a)
    t = np.arange(0, 1, dt*div)
    if len(t) <= 2:
        pdb.set_trace()
    return spl(t)


def tsp_glyph(n, r1=0.5, r2=1.0, center=[0,0]):
    P = geom.shapes.random_radial_polygon(n, r1, r2, center=center)
    I = inverse_path_tsp(P)
    return P[I,:]


def smooth_strokes_lqt(S, only_pos=True, penups=True, penup_variability=0.4, **kwargs):
    trajectories = []
    for P in S:
        if penups:
            variability = [1, penup_variability] + [1 for _ in P[1:-1]] + [penup_variability, 1]
        else:
            variability = []
        trajectory = smooth_stroke_lqt(P, variability=variability, **kwargs)
        trajectories.append(trajectory)
    return trajectories

def smooth_stroke_lqt(X, only_pos=True, **kwargs):
    import motor.lqt as lqt
    #return st.smooth_stroke(X, ds=1, degree=1)
    dim  = X.shape[1]
    default_params = dict(angle=0,
                          sigma=1.0,
                          order=3,
                          opt_x0=True,
                          stop_motion=True,
                          isotropy=1,
                          activation_sigma=0.5,
                          subd=5,
                          subsample=100) # Assuming duration of 0.5
    default_params.update(kwargs)
    X = lqt.semitied_lqt(X.T, dim=dim, **default_params)
    if only_pos:
        return X[:dim, :].T
    return X.T


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
    print('dt_pre:', u[1] - u[0])
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
