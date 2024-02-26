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
q_min_braccio = [0.0, 0.262, 0.0, 0.0, 0.0]
q_max_braccio = [np.pi, 2.879, np.pi, np.pi, np.pi]
braccio_limits = [q_min_braccio, q_max_braccio]

# Franka emika joint limits
q_min_panda = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
q_max_panda = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])
panda_limits = [q_min_panda, q_max_panda]

def is_number(x):
    return isinstance(x, numbers.Number)

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
                 tip_frame='tiba_pen_tcp',
                 subsample=10):
        self.q0 = q0
        self.dq0 = np.zeros(len(q0))
        self.rbt = KDLRobot(urdf_path, base_frame, tip_frame, self.q0, self.dq0)
        self.subsample = subsample

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

        config.state.progress = 0

        for i, traj in enumerate(trajectories):
            if len(traj) < 2:
                continue
            approach = tu.trapezoidal_approach(x0, traj[0])
            if len(approach) > 1:
                linked_trajectories.append(tu.rest_inout(approach, rest*0.5))
            if rest > 0.0:
                # TODO double check rest. Seems to give some acceleration artefacts
                traj = tu.rest_inout(traj, rest*0.5)
                # dX = np.diff(traj, axis=0)/cfg.dt
                # speed = np.sqrt(np.sum(dX**2, axis=1))
                # try:
                #     if np.max(speed) > 2:
                #         pdb.set_trace()
                #         print('Oh shit')
                # except:
                #     pdb.set_trace()
                #     print('ooooo')
            x0 = traj[-1]
            #break
            linked_trajectories.append(traj)


        all_trajs = np.vstack(linked_trajectories)

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
                # Add velocities appended to first control command
                dQs.append([np.zeros(nb_control_vars)] + dQs_batch)
                q0 = Qs_batch[-1]

            except Exception as e:
                print(e)
            config.state.progress = tindex+1

        Qs = sum(Qs, [])
        dQs = sum(dQs, [])
        Qs_1 = np.array(Qs)
        dQs_1= np.array(dQs)

        if self.subsample > 1:
            n = len(Qs)
            Qs, dQs = tu.bspline((n-1)*self.subsample+1, Qs, der=1, duration=(n-1)*cfg.dt)
            print('Splined shapes', Qs.shape, dQs.shape)

        Qs = np.array(Qs)
        dQs = np.array(dQs)

        return list(Qs), list(dQs)


def plot_trajectory_stats(robot, Qs, dQs, trajectories=[], figsize=(13, 13), dpi=92, max_segments=200, filename=''):
    fig = ply.figure(title='Joint and task trajectory')

    Qs = np.array(Qs)
    dQs = np.array(dQs)

    # Plot joint positions
    ply.subplot(1, 2)
    ply.title('joint positions (rad)')
    # Speed is in radians/sec

    for i in range(7):
        q = Qs[:, i]
        #dq = np.diff(Qs[:,i])
        ply.plot(q, label='q%d'%i)
    #plt.legend(loc='right')

    # Plot joint speeds
    ply.subplot(2, 2)
    ply.title('joint speed deg/sec')
    # Speed is in radians/sec
    for i in range(7):
        dq = dQs[:, i]
        #dq = np.diff(Qs[:,i])
        ply.plot(geom.degrees(dq)) #, color=ply.default_color(i))

    limit = geom.degrees(cfg.default_angular_velocity)
    ply.rectangle([0, -limit], [len(dQs), limit], color='k', alpha=0.1, fill=True)

    ply.subplot(1, 3)
    # Joint limit graph
    ply.title('Joints min/max')
    ply.bar([np.min(Qs, axis=0), np.max(Qs, axis=0)], limits=[q_min_panda, q_max_panda])

    pts = np.array([robot.pen_tcp_pos(q)[:3] for q in Qs])
    vel = np.diff(pts, axis=0)/cfg.robot_dt
    acc = np.diff(pts, n=2, axis=0)/(cfg.robot_dt**2)

    ply.subplot(1, 1, title='Cartesian velocity (m/s)')
    for i in range(3):
        t = np.linspace(0, len(vel), len(vel))*cfg.robot_dt
        ply.plot(t, vel[:,i], ['r', 'g', 'b'][i], label=['x', 'y', 'z'][i], alpha=0.7)
    ply.plot(t, tu.rownorm(vel), 'k', lw=2, label='norm')

    limit = cfg.default_cartesian_speed
    ply.rectangle([0, -limit], [t[-1], limit], color='k', alpha=0.1, fill=True)

    ply.subplot(2, 1, title='Cartesian acceleration (m/s^2)')
    for i in range(3):
        t = np.linspace(0, len(acc), len(acc))*cfg.robot_dt
        ply.plot(t, acc[:,i], ['r', 'g', 'b'][i], alpha=0.7)
    ply.plot(t, tu.rownorm(acc), 'k', lw=2)

    limit = cfg.default_cartesian_speed / cfg.acceleration_time
    ply.rectangle([0, -limit], [t[-1], limit], color='k', alpha=0.1, fill=True)

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
