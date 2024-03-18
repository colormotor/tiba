'''
o-O-o o-O-o o--o    O
  |     |   |   |  / \
  |     |   O--o  o---o
  |     |   |   | |   |
  o   o-O-o o--o  o   o

Painting robot utils.
© Daniel Berio (@colormotor) 2023 - ...

ilqr - Iterative Linear Quadratic Regulator utilities for robot inverse kinematics
Mostly wraps the Idiap iLQR implementation from Idiap Research Institute
https://github.com/idiap/ilqr_planner

'''
from importlib import reload
import os, sys
import pdb
import numbers
import platform, subprocess
import numpy as np
import matplotlib.pyplot as plt

# Using Idiap iLQR implementation for inverse kinematics planning
from PyLQR.system import PosOrnPlannerSys,PosOrnKeypoint
from PyLQR.solver import ILQRRecursive
from PyLQR.sim import KDLRobot
from PyLQR.utils import primitives, PythonCallbackMessage

from scipy.interpolate import BPoly

from . import utils, geom, config
from . import trajectory_utils as tu
from .config import cfg
from . import draw
from . import matplotlyb as ply

from spatialmath import SE3, SO3, UnitQuaternion
import roboticstoolbox as rtb
from spatialmath.base import trnorm
from tqdm import tqdm, trange

reload(ply)

# Braccio joint limits
# q_min_braccio = [0.0, 0.262, 0.0, 0.0, 0.0]
# q_max_braccio = [np.pi, 2.879, np.pi, np.pi, np.pi]
# braccio_limits = [q_min_braccio, q_max_braccio]

# Franka emika joint limits
# q_min_panda = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
# q_max_panda = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])
# panda_limits = [q_min_panda, q_max_panda]

def is_number(x):
    return isinstance(x, numbers.Number)

def batch_indices(n, batch_size):
    n_batches = n//batch_size + 1
    if n_batches < 2:
        return [np.array(list(range(n)))]
    I = np.linspace(0, n-1, n_batches).astype(int)
    return [np.array(list(range(a, b+1))) for a, b in zip(I, I[1:])]

def check_limits(robot, q):
    return np.all(q >= robot.q_min, axis=0) and np.all(q <= robot.q_max, axis=0)

def subsample_polynomial(derivs, subsample=10):
    derivs = np.dstack(derivs)
    dim = derivs.shape[1] #len(derivs[0][0])
    nderiv = derivs.shape[2]
    n = derivs.shape[0]
    polys = []
    dpolys = []
    T = np.linspace(0, (n-1)*cfg.dt, n)
    #T = np.linspace(0, (n-1), n)
    try:
        for d in range(dim):
            polys.append(BPoly.from_derivatives(T, np.vstack([derivs[:, d, i] for i in range(nderiv)]).T))
            dpolys.append(polys[-1].derivative())
    except Exception as e:
        print(e)
        raise(ValueError)
    T = np.arange(0, T[-1], cfg.robot_dt) # np.linspace(0, T[-1], (n-1)*subsample+1)
    X = np.vstack([bp(T) for bp in polys]).T
    dX = np.vstack([bp(T) for bp in dpolys]).T
    return list(X), list(dX)

dtype = np.float64
class iLQRPlannerVelocity:
    def __init__(self, urdf_path, q0,
                 base_frame='tiba_link0',
                 tip_frame='tiba_pen_tcp',
                 subsample=10):
        self.q0 = q0.astype(dtype)
        self.dq0 = np.zeros(len(q0)).astype(dtype)
        self.rbt = KDLRobot(urdf_path, base_frame, tip_frame, self.q0, self.dq0)
        self.subsample = subsample

    def plan(self, trajectories,
             num_iter=20,
             r=1e-5,
             weights=np.ones(7),
             rest = 0.1,
             limits=[],
             get_trajectories=False):
        if (is_number(trajectories[0][0])): # Single trajectory case
            trajectories = [trajectories]
        Q = np.diag(weights).astype(dtype)
        trajectories = list(trajectories)
        # Add approaches between trajectories
        x0 = tu.to_pos_orn(tu.to_pose(np.concatenate([self.rbt.get_ee_pos(), self.rbt.get_ee_orn()])))

        print('Initial position (KDL)', tu.to_pose(x0))
        print('Initial trajectory position ', tu.to_pose(trajectories[0][0]))
        print('x0', x0, 'x0_traj', trajectories[0][0])
        print('rest is', rest)

        linked_trajectories = []

        config.state.progress = 0

        cartesian_approaches = True

        rest = 0.0
        for i, traj in enumerate(trajectories):
            if len(traj) < 2:
                continue
            if cartesian_approaches:
                approach = tu.trapezoidal_approach(x0, traj[0])
                if len(approach) > 1:
                    linked_trajectories.append(tu.rest_inout(approach, rest))

            linked_trajectories.append(tu.rest_inout(traj, rest))
            x0 = traj[-1]

        # linked_trajectories = [np.vstack([np.array(P).astype(dtype) for P in linked_trajectories])]
        # linked_trajectories += [tu.rest_inout(trajectories[0], rest*0.5)]
        # linked_trajectories = [np.array(traj).astype(dtype) for traj in linked_trajectories]
        # Plan
        nb_control_vars = len(self.dq0)
        Qs = []
        dQs = []
        u0_t = np.zeros(nb_control_vars).astype(dtype)

        q0 = self.q0
        dq0 = self.dq0

        for tindex, states in enumerate(linked_trajectories):

            print('Planning trajectory %d of %d'%(tindex+1, len(linked_trajectories)))
            timesteps = list(range(len(states)))
            self.rbt.set_conf(q0, dq0, True)
            start = 0
            keypoints = [PosOrnKeypoint(state[:3], state[3:], Q, i+start) for i, state in enumerate(states)]
            horizon = len(timesteps)+start
            cmd_penalties = list(r*np.ones(nb_control_vars))

            if False: #limits:
                sys = PosOrnPlannerSys(self.rbt, keypoints, cmd_penalties, limits[1], limits[0], horizon, 1, cfg.dt)
            else:
                sys = PosOrnPlannerSys(self.rbt, keypoints, cmd_penalties, horizon, 1, cfg.dt)

            print('cart d', np.diff(np.array(states), axis=0)[:10])
            planner = ILQRRecursive(sys)
            u0_t = np.zeros(nb_control_vars).astype(dtype)
            u0 = np.tile(u0_t, horizon-1)

            cb = PythonCallbackMessage()

            try:
                Q_batch, _, dQ_batch, _, _ = planner.solve(
                    u0.reshape((-1, nb_control_vars)),
                    num_iter,
                    True, True, cb)

                #print("Shapes, ", len(Q_batch), len(dQ_batch))
                #dQ_batch = list(np.diff(np.array(Q_batch), axis=0)/cfg.dt)

                print('Dq_ends', dQ_batch[1], dQ_batch[-1])

                # The iLQR optimizer seems to create non zero velocities at the start of the trajectory
                # hacky fix

                # #
                # dQ_approach = dQ_batch + [dQ_batch[-1] + (np.zeros(7) - dQ_batch[-1])*t for t in np.linspace(0, 1, rep)]
                # Q_approach = Q_approach + [np.array(Q_approach[-1])]*rep #

                #Q_batch = [Q_batch[0]]*rep + Q_batch + [Q_batch[-1]]*rep
                #dQ_batch = [np.zeros(nb_control_vars)]*rep + dQ_batch + [np.zeros(nb_control_vars)]*rep
                print("Before subsample")

                acc = np.diff(np.array(dQ_batch), axis=0)/cfg.dt
                if np.any(np.abs(acc) > 4):
                    print('High acceleration inside trajectory')
                    print(acc)
                    #pdb.set_trace()
                print('Max accel local:', np.max(np.abs(acc)))

                n = len(Q_batch)
                Q_batch, dQ_batch = tu.bspline((n-1)*self.subsample+1, Q_batch, der=1, duration=(n-1)*cfg.dt)
                Q_batch = list(Q_batch)
                dQ_batch = list(dQ_batch)

                rep = 50
                dQ_approach = [dq0 + (dQ_batch[0] - dq0)*t for t in np.linspace(0, 1, rep)]
                Q_approach = [q0 + (Q_batch[0] - q0)*t for t in np.linspace(0, 1, rep)]

                # print('lens', len(dQ_batch), len(Q_batch))
                dQ_batch = dQ_approach + dQ_batch
                Q_batch = Q_approach + Q_batch # One extra sample for positions


                dQ_batch = dQ_batch + [dQ_batch[-1] + (np.zeros(7) - dQ_batch[-1])*t for t in np.linspace(0, 1, rep)]
                Q_batch = Q_batch + [np.array(Q_batch[-1])]*rep #


                # print("After subsample")
                # acc = np.diff(np.array(dQ_batch), axis=0)/cfg.robot_dt
                # if np.any(np.abs(acc) > 4):
                #     print('High acceleration inside trajectory')
                #     print(acc)
                #     #pdb.set_trace()
                # print('Max accel local:', np.max(np.abs(acc)))


                Qs.append(Q_batch)
                dQs.append(dQ_batch)

                # Qs, dQs = subsample_polynomial([Qs, dQs, ddQs], self.subsample)
                # Qs.append(Qs_batch)
                # # Add velocities appended to first control command
                # dQs.append([np.zeros(nb_control_vars)] + dQs_batch)



                # rep = 5
                # Qs[-1] = [Qs[-1][0]]*rep + Qs[-1] + [Qs[-1][-1]]*rep
                # dQs[-1] = [np.zeros(7)]*rep + dQs[-1] + [np.zeros(7)]*rep
                # #ddQs[-1] = [np.zeros(7)]*rep + ddQs[-1] + [np.zeros(7)]*rep

                print('q0 to start', q0 - Q_batch[0])
                q0 = Q_batch[-1]

            except Exception as e:
                print(e)
            config.state.progress = tindex+1


        # dq0 = self.dq0
        # Qs_joined = []
        # dQs_joined = []

        # q0 = self.q0
        # dq0 = self.dq0
        # Qs_joined = []
        # dQs_joined = []
        # for Q, dQ in zip(Qs, dQs):
        #     Qapproach = [q0 + t*(Q[0] - q0) for t in np.linspace(0, 1, 10)]

        #     if Qapproach:
        #         print('Join approach', len(Qapproach))
        #         Qs_joined.append(Qapproach)
        #         dQs_joined.append(dQapproach)
        #     Qs_joined.append(Q)
        #     dQs_joined.append(dQ)

        #     Qapproach, dQapproach = tu.joint_trajectory_time_optimal([q0, Q[0]], dt=cfg.dt)
        #     if Qapproach:
        #         print('Join approach', len(Qapproach))
        #         Qs_joined.append(Qapproach)
        #         dQs_joined.append(dQapproach)
        #     Qs_joined.append(Q)
        #     dQs_joined.append(dQ)
        #     q0 = Q[-1]

        Qs = sum(Qs, [])
        dQs = sum(dQs, [])
        #Qs = sum(Qs, [])
        #dQs = sum(dQs, [])
        # if self.subsample > 1:
        #     n = len(Qs)
        #     Qs, dQs = tu.bspline((n-1)*self.subsample+1, Qs, der=1, duration=(n-1)*cfg.dt)
        #     print('Splined shapes', Qs.shape, dQs.shape)
        if get_trajectories:
            return Qs, dQs, linked_trajectories
        return Qs, dQs


class iLQRPlannerAcceleration:
    def __init__(self, urdf_path, q0,
                 base_frame='tiba_link0',
                 tip_frame='tiba_pen_tcp',
                 subsample=10):
        self.q0 = q0.astype(dtype)
        self.dq0 = np.zeros(len(q0)).astype(dtype)
        self.ddq0 = np.zeros(len(q0)).astype(dtype)
        self.rbt = KDLRobot(urdf_path, base_frame, tip_frame, self.q0, self.dq0)
        self.subsample = subsample

    def plan(self, trajectories,
             num_iter=20,
             r=1e-5,
             weights=np.ones(6),
             rest = 0.1,
             limits=[],
             qd_lim=None,
             get_trajectories=False):
        if (is_number(trajectories[0][0])): # Single trajectory case
            trajectories = [trajectories]
        Qtarget = np.zeros((12, 12)).astype(dtype)
        Qtarget[:6,:6] = np.diag(weights).astype(dtype)

        trajectories = list(trajectories)
        # Add approaches between trajectories
        x0 = np.concatenate([self.rbt.get_ee_pos(), self.rbt.get_ee_orn()])
        print('Initial position (KDL)', tu.to_pose(x0))
        print('Initial trajectory position ', tu.to_pose(trajectories[0][0]))

        linked_trajectories = []

        config.state.progress = 0

        cartesian_approaches = True
        rest = 0.0
        for i, traj in enumerate(trajectories):
            if len(traj) < 2:
                continue
            if cartesian_approaches:
                approach = tu.trapezoidal_approach(x0, traj[0])
                if len(approach) > 1:
                    linked_trajectories.append(approach)

            linked_trajectories.append(tu.rest_inout(traj, rest))
            x0 = traj[-1]


        # Plan
        nb_control_vars = len(self.dq0)
        Qs = []
        dQs = []
        ddQs = []
        u0_t = np.zeros(nb_control_vars).astype(dtype)

        q0 = self.q0
        dq0 = self.dq0
        for tindex, states in enumerate(linked_trajectories):
            states = np.array(states)
            states = np.hstack([states, np.zeros_like(states)])

            Qtargets = [np.array(Qtarget) for _ in states]
            for i in range(1):
                Qtargets[i][6:,6:] = np.eye(6)*100
                Qtargets[-1-i][6:,6:] = np.eye(6)*100
            print('Planning trajectory %d of %d'%(tindex+1, len(linked_trajectories)))
            print('States shape:', states.shape)
            timesteps = list(range(len(states)))
            self.rbt.set_conf(q0, dq0, True)
            keypoints = [PosOrnKeypoint(states[ind][:3], # pos
                                        states[ind][7:7+3], # pos vel
                                        states[ind][3:7], # orn
                                        states[ind][7+3:], # orn vel,
                                        Qtargets[ind],
                                        i) for i, ind in enumerate(timesteps)]
            horizon = len(timesteps)
            cmd_penalties = list(r*np.ones(nb_control_vars))

            if limits:
                sys = PosOrnPlannerSys(self.rbt, keypoints, cmd_penalties, limits[1], limits[0], qd_lim, -qd_lim, horizon, 2, cfg.dt)
            else:
                sys = PosOrnPlannerSys(self.rbt, keypoints, cmd_penalties, horizon, 2, cfg.dt)

            print("Creating planner")
            planner = ILQRRecursive(sys)

            u0 = np.tile(u0_t, horizon-1)
            cb = PythonCallbackMessage()

            print("PLanning")
            try:
                QdQs_batch, _, ddQs_batch, _, _ = planner.solve(
                    u0.reshape((-1, nb_control_vars)),
                    num_iter,
                    True, True, cb)

                QdQs_batch = np.array(QdQs_batch)
                print('Result shape', QdQs_batch.shape)
                Qs.append(list(QdQs_batch[:, :7]))
                dQs.append(list(QdQs_batch[:, 7:]))
                ddQs.append([np.zeros(7)] + ddQs_batch)

                # rep = 5
                # Qs[-1] = [Qs[-1][0]]*rep + Qs[-1] + [Qs[-1][-1]]*rep
                # dQs[-1] = [np.zeros(7)]*rep + dQs[-1] + [np.zeros(7)]*rep
                # ddQs[-1] = [np.zeros(7)]*rep + ddQs[-1] + [np.zeros(7)]*rep
                print('accel size', np.array(ddQs_batch).shape)

                q0 = QdQs_batch[-1][:7]

            except Exception as e:
                print(e)
            config.state.progress = tindex+1


        # q0 = self.q0
        # Qs_joined = []
        # dQs_joined = []
        # for Q, dQ, ddQ in zip(Qs, dQs, ddQs):
        #     Qapproach, dQapproach = tu.joint_trajectory_time_optimal([q0, Q[0]], dt=cfg.dt)
        #     if Qapproach:
        #         print('Join approach', len(Qapproach))
        #         Qs_joined.append(Qapproach)
        #         dQs_joined.append(dQapproach)
        #     Qs_joined.append(Q)
        #     dQs_joined.append(dQ)
        #     q0 = Q[-1]

        Qs = sum(Qs, [])
        dQs = sum(dQs, [])
        ddQs = sum(ddQs, [])
        #dQs = sum(dQs_joined, [])
        #Qs = sum(Qs, [])
        #dQs = sum(dQs, [])
        #Qs_1 = np.array(Qs).astype(dtype)
        #dQs_1= np.array(dQs).astype(dtype)
        #ddQs_1 = np.array(ddQs).astype(dtype)
        if self.subsample > 1:
            n = len(Qs)
            print('Len item', len(ddQs[0]))
            #print(np.array(Qs).shape, np.array(dQs).shape, np.array(ddQs).shape)
            Qs, dQs = subsample_polynomial([Qs, dQs, ddQs], self.subsample)
            #Qs, dQs = tu.bspline((n-1)*self.subsample+1, Qs, der=1, duration=(n-1)*cfg.dt)
            print('Splined shapes', Qs.shape, dQs.shape)

        if get_trajectories:
            return Qs, dQs, linked_trajectories
        return Qs, dQs

        Qs = np.array(Qs).astype(dtype)
        dQs = np.array(dQs).astype(dtype)

        return list(Qs), list(dQs)


def plot_trajectory_stats(robot, Qs, dQs, trajectories=[], figsize=(13, 13), dpi=92, max_segments=200, filename=''):
    fig = ply.figure(title='Joint and task trajectory')

    Qs = np.array(Qs)
    dQs = np.array(dQs)

    # Plot joint positions
    ply.subplot(1, 2)
    ply.title('joint positions (rad)')
    # Speed is in radians/sec

    t = np.linspace(0, cfg.robot_dt*(len(Qs)-1), len(Qs))
    for i in range(7):
        q = Qs[:, i]
        #dq = np.diff(Qs[:,i])
        ply.plot(t, q, label='q%d'%i)
    #plt.legend(loc='right')

    # Plot joint speeds
    ply.subplot(2, 2)
    ply.title('joint speed rad/sec')
    t = np.linspace(0, cfg.robot_dt*(len(Qs)-1), len(dQs))
    # Speed is in radians/sec
    for i in range(7):
        dq = dQs[:, i]
        #dq = np.diff(Qs[:,i])
        ply.plot(t, dq) #geom.degrees(dq)) #, color=ply.default_color(i))

    n = len(dQs)
    for i, l in enumerate(robot.qd_lim):
        ply.line([0, -l], [t[-1], -l], ply.categorical_color(i), ls=':')
        ply.line([0, l], [t[-1], l], ply.categorical_color(i), ls=':')

    #limit = cfg.default_joint_speed
    #ply.rectangle([0, -limit], [len(dQs), limit], color='k', alpha=0.1, fill=True)

    ply.subplot(1, 3)
    # Joint limit graph
    ply.title('Joints min/max')
    ply.bar([np.min(Qs, axis=0), np.max(Qs, axis=0)], limits=[robot.q_min, robot.q_max])

    pts = np.array([robot.pen_tcp_pos(q)[:3] for q in Qs])
    vel = np.diff(pts, axis=0)/cfg.robot_dt
    acc = np.diff(pts, n=2, axis=0)/(cfg.robot_dt**2)

    ply.subplot(1, 1, title='Cartesian velocity (m/s)')
    for i in range(3):
        t = np.linspace(0, len(vel), len(vel))*cfg.robot_dt
        ply.plot(t, vel[:,i], ['r', 'g', 'b'][i], label=['x', 'y', 'z'][i], alpha=0.7)
    #ply.plot(t, tu.rownorm(vel), 'k', lw=2, label='norm')
    #t = np.linspace(0, len(acc), len(acc))*cfg.robot_dt
    #ply.plot(t, acc[:,0], 'k', alpha=0.7)

    limit = cfg.default_cartesian_speed
    ply.rectangle([0, -limit], [t[-1], limit], color='k', alpha=0.1, fill=True)

    ply.subplot(2, 1, title='Joint acceleration (rad/s^2)')
    acc = np.diff(dQs, axis=0)/cfg.robot_dt
    for i in range(7):
        t = np.linspace(0, len(acc), len(acc))*cfg.robot_dt
        ply.plot(t, acc[:,i]) #, ['r', 'g', 'b'][i], alpha=0.7)

    limit = robot.qdd_lim[0]
    ply.rectangle([0, -limit], [t[-1], limit], color='k', alpha=0.1, fill=True)

    # ply.subplot(2, 1, title='Cartesian acceleration (m/s^2)')
    # for i in range(1,2):
    #     t = np.linspace(0, len(acc), len(acc))*cfg.robot_dt
    #     ply.plot(t, acc[:,i], ['r', 'g', 'b'][i], alpha=0.7)
    # #ply.plot(t, tu.rownorm(acc), 'k', lw=2)

    #limit = cfg.default_cartesian_speed / cfg.acceleration_time
    #ply.rectangle([0, -limit], [t[-1], limit], color='k', alpha=0.1, fill=True)

    ply.subplot(2, 3)
    box = geom.shapes.box_3d([-0.8, -0.8, 0], [1, 1, 1.2])
    ply.stroke3d(box, color='r', lw=1, ls=':', alpha=0.25)

    if trajectories:
        for trajectory in trajectories:
            trajectory = np.array(trajectory)
            ply.plot3d(trajectory, 'r', lw=2, alpha=1)

    ply.plot3d(pts, 'b', lw=2, alpha=0.7)

    I = np.linspace(0, len(pts)-1, max_segments).astype(int)
    for i in I:
        q = Qs[i]
        P, penP = robot.robot_polylines(q)
        P = np.array(P)
        ply.plot3d(P, color=ply.cmap(i/I[-1]), alpha=0.2)

    ply.show()


def show_open(filename):
    plt.savefig(filename)
    if platform.system() == 'Darwin':
        subprocess.run(['open', filename])
    else:
        subprocess.run(['xdg-open', filename])
