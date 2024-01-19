#!/usr/bin/env python3
import numpy as np
from importlib import reload
import importlib

from motor import lqt
import imgui
from polygonsoup import utils, geom, svg, bezier, plut
import roboticstoolbox as rtb
import pdb
import time

from .interface import TibaInterface
from .robot import TibaRobot
from .draw import View3d, SketchView3d, GlView3d
from . import planner
from .config import cfg
from . import trajectory_utils as tu
from .gfx import renderer
reload(renderer)
reload(tu)
reload(planner)
from spatialmath import SE3, SO3, UnitQuaternion
from spatialmath.base import trnorm
import roboticstoolbox as rtb
from collections import namedtuple
import matplotlib.pyplot as plt
import swift
import matplotlib

state = lambda: None
state.has_ros = False

rosloader = importlib.find_loader('rospy')

if rosloader is not None:
    print("Loading ros")
    import rospy
    from std_msgs.msg import String
    from trajectory_msgs.msg import JointTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint
    from std_msgs.msg import Float64MultiArray
    from std_msgs.msg import Time
    from std_msgs.msg import Header
    from std_msgs.msg import Duration
    state.has_ros = True

def plan_samples(draft, max_speed, accel_time, penup_distance, debug=True):
    print(draft)
    print(max_speed)
    print(accel_time)
    print(penup_distance)

    paths = tu.plan_polylines_jerk(draft, 0.1, max_speed=max_speed, max_accel_time=accel_time, maxiter=300)
    paths = [np.hstack([P, np.zeros((len(P), 1))]) for P in paths]
    lifts = [False]
    trajectories = [paths[0]]
    if len(paths) > 1:
        for P1, P2 in zip(paths, paths[1:]):
            if geom.distance(P1[-1], P2[0]) > 2e-3:
                lifts.append(True)
                trajectories.append(tu.lifting_motion(P1[-1], P2[0], penup_distance, max_accel_time=1.0))
            lifts.append(False)
            trajectories.append(P2)

    if debug:
        fig = plt.figure(figsize=((12, 4)))
        plt.subplot(121)
        T = 0.0
        for X, lift in zip(trajectories, lifts):
            speed = tu.compute_speed(X, cfg.dt)
            t = np.linspace(0, len(speed)*cfg.dt, len(speed)) + T
            clr = 'c' if lift else 'r'
            plt.plot(t, speed, clr, lw=1)
            T = t[-1] + cfg.dt
        ax = fig.add_subplot(224, projection='3d')
        for i, X in enumerate(trajectories):
            plt.plot(X[:,0], X[:,1], X[:,2], color=plut.default_color(i), lw=1, alpha=1)
        plt.tight_layout()
        planner.show_open('cart.pdf')
    return trajectories

def plan_bezier(draft, max_speed, accel_time, penup_distance, debug=True):
    print(draft)
    print(max_speed)
    print(accel_time)
    print(penup_distance)

    paths = tu.plan_bezier_chains_jerk(draft, max_speed=max_speed, max_accel_time=accel_time, maxiter=300)
    paths = [np.hstack([P, np.zeros((len(P), 1))]) for P in paths]
    lifts = [False]
    trajectories = [paths[0]]
    if len(paths) > 1:
        for P1, P2 in zip(paths, paths[1:]):
            if geom.distance(P1[-1], P2[0]) > 2e-3:
                lifts.append(True)
                trajectories.append(tu.lifting_motion(P1[-1], P2[0], penup_distance, max_accel_time=1.0))
            lifts.append(False)
            trajectories.append(P2)

    if debug:
        fig = plt.figure(figsize=((12, 4)))
        plt.subplot(121)
        T = 0.0
        for X, lift in zip(trajectories, lifts):
            speed = tu.compute_speed(X, cfg.dt)
            t = np.linspace(0, len(speed)*cfg.dt, len(speed)) + T
            clr = 'c' if lift else 'r'
            plt.plot(t, speed, clr, lw=1)
            T = t[-1] + cfg.dt
        ax = fig.add_subplot(224, projection='3d')
        for i, X in enumerate(trajectories):
            plt.plot(X[:,0], X[:,1], X[:,2], color=plut.default_color(i), lw=1, alpha=1)
        plt.tight_layout()
        planner.show_open('cart.pdf')
    return trajectories

dip_pos = np.array([0.11812825731443674, -0.407704314564537, 0.10439764358100054,
                    -0.008107937414934388, 0.8524478856525898, -0.5214394134973193, -0.036983802520516196])

#  [-0.7743211365357173, -0.10300303344311892, -0.50291393240498, -2.5627437008673333, -0.1284246296894626, 2.401976760003019, 0.7079839864811134]
#
class TibaSketch:
    def __init__(self):

        self.clear()

        self.params_to_check = [
            'seed',
            'drawing.padding',
            'drawing.penup_mm',
            'drawing.pos_offset_x',
            'drawing.pos_offset_y',
            'traj.sigma',
            'traj.isotropy',
            'traj.angle'
        ]

    def init(self, sketch, robot, callback, panda=None, gripper=None):
        self.itiba = TibaInterface(robot, panda=panda, gripper=gripper)
        self.v3d = GlView3d(sketch)
        self.robot = robot
        self.pad = self.itiba.pad
        self.max_steps = 500
        self.max_trajectory_points = 200
        self.sketch = sketch

        self.trajectory_callback = callback
        # self.robot.q = self.robot.qr
        # self.robot.q = np.array([-0.7743211365357173, -0.10300303344311892, -0.50291393240498, -2.5627437008673333, -0.1284246296894626, 2.401976760003019, 0.7079839864811134])
        self.robot.q = self.robot.home

        if state.has_ros:
            try:
                self.msg_pub = rospy.Publisher('/tiba/msg', String, queue_size=10)
                self.jtraj_pub = rospy.Publisher('/tiba/jtraj', JointTrajectory, queue_size=10)
                rospy.Subscriber('/tiba/q', JointTrajectoryPoint, self.state_callback)
                rospy.Subscriber('/tiba/response', String, self.response_callback)
                rospy.init_node('motion_generator', log_level=rospy.ERROR, anonymous=True)
                self.send_msg('start: Tiba sketch connecting')
            except rospy.exceptions.ROSException as e:
                print("Ros issue")
                state.has_ros = False

    def send_msg(self, data):
        print("Sending msg", data)
        self.msg_pub.publish(data)


    def response_callback(self, data):
        print("Received response: ", data.data)

    def state_callback(self, data):
        print("Received state")
        self.robot.q = np.array(data.positions)
        #self.itiba.Qs = []
        #self.steps, self.fk_points = [], []
        #self.trajectories_computed = False
        #self.trajectories = []

    @staticmethod
    def add_parameters(params_to_update):
        params = {
            'Drawing':  {
              'Plane offset': (0.0, {'min':-20,'max': 20}),
              'Padding': (80, {'min':0.0, 'max':450}),
              'Penup mm': (50, {'min':0.0, 'max':200}),
              'Pos offset x': (0.0, {'min': -1, 'max': 1}),
              'Pos offset y': (0.0, {'min': -1, 'max': 1}),
            },
            'Tsm': {
              'Dist': (4.0, {'min':0, 'max':10}),
              'Xoffset': (0.0, {'min':-1, 'max':1}),
              'Voffset': (0.0, {'min':-1, 'max':1}),
              'rotx': (275, {'min':0, 'max':360}),
              'roty': (0, {'min':0, 'max':360}),
              'rotz': (138, {'min':0, 'max':360}),
          }
        }
        params_to_update.update(params)

    @property
    def c(self):
        return self.sketch.canvas

    def set_drawing_data(self, data):
        self.drawing_data = data

    def compute_steps(self, Qs):
        max_steps = self.max_steps
        n = min(len(Qs), max_steps)
        I = np.linspace(0, len(Qs)-1, n, dtype=int)
        polylines = []
        points = []

        for i in I:
            q = Qs[i]
            P, penP = self.robot.robot_polylines(q)
            polylines.append(P)
            polylines.append(penP)
            points.append(self.robot.pen_tcp_pos(q)[:3])
        return polylines, points

    def draw_trajectory_2d(self, P):
        max_steps = self.max_trajectory_points
        n = min(len(P), max_steps)
        I = np.linspace(0, len(P)-1, n, dtype=int)
        P = np.array(P)[I]
        self.c.polyline(P)

    def draw_trajectory_3d(self, P):
        max_steps = self.max_trajectory_points
        n = min(len(P), max_steps)
        I = np.linspace(0, len(P)-1, n, dtype=int)
        P = np.array(P)[I]
        self.v3d.polyline(P)

    def send_joint_trajectory(self, Qs, dQs):
        jtraj = JointTrajectory()
        jtraj.header = Header()
        jtraj.header.stamp = rospy.Time.now()
        jtraj.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6',
                                'panda_joint7']
        for i, (q, dq) in enumerate(zip(Qs, dQs)):
            p = JointTrajectoryPoint()
            p.positions = q
            p.velocities = dq
            #p.time_from_start = Duration(0, int(1e+9*cfg.robot_dt*i))
            jtraj.points.append(p)
        self.jtraj_pub.publish(jtraj)
        rospy.loginfo("Sent joint trajectory")

    def params_changed(self, *args):
        for key in args:
            if key in self.sketch.gui.changed:
                return True
        return False

    def clear(self):
        self.steps = []
        self.fk_points = []
        self.draft = []
        self.trajectories = []
        self.dirty = True
        self.trajectories_computed = False

    def set_draft(self, draft):
        self.clear()
        self.draft = draft

    def dip_trajectory(self):
        up = np.array([0, 0, 1, 0, 0, 0, 0])
        start_offset = 70/1000
        dip_offset = 35/1000
        clean_offset = 10/1000
        dip_start = dip_pos + up*start_offset

        print(dip_start)
        dip_start = planner.align_pos_z(dip_start, np.array([0, 0, -1]))
        dip_rim = dip_start - up*start_offset
        dip_in = dip_rim - up*dip_offset #planner.align_frame_z(dip_start, np.array([0, 0, -1]))
        dip_clean = dip_rim - up*clean_offset
        cur_pos = self.robot.pen_tcp_pos(self.robot.q)
        trajectories = [tu.approach_sequence([cur_pos, dip_start, dip_in])]
        trajectories += [tu.xy_circles_mm(dip_in, 7.5, 1.5, 4, ease_duration=1.0)]
        trajectories += [tu.approach_sequence([trajectories[-1][-1], dip_clean])] #, cur_pos])]
        trajectories += [tu.xy_circles_mm(dip_clean, 14, 1.5, 4, ease_duration=1.0)]
        trajectories += [tu.approach_sequence([trajectories[-1][-1], dip_start])] #, cur_pos])]
        # trajectories = [tu.trapezoidal_approach(cur_pos, dip_start),
        #                 tu.trapezoidal_approach(dip_start, dip_next),
        #                 tu.xy_circles_mm(dip_next, 10, 1.5, 4, ease_duration=1.0),
        #                 tu.trapezoidal_approach(dip_next, dip_start)]

        self.itiba.plan_task_trajectories(trajectories)
        self.steps, self.fk_points = self.compute_steps(self.itiba.Qs)
        self.trajectories_computed = True
        print('Dip trajectory')

    def compute_trajectories(self, max_speed=250/1000, accel_time=0.4):
        params = self.sketch.params.params
        if not self.draft:
            print("No draft")
            return

        callback = self.trajectory_callback
        if type(callback)==str:
            if callback=='bezier':
                callback = plan_bezier
            elif callback=='samples':
                callback = plan_samples
            elif callback=='lqt':
                callback = plan_lqt
            else:
                raise ValueError('Unknown callback')
        draft = self.pad.fit_to_box(self.draft, padding=params.drawing.padding, penup_mm=0.0)
        trajectories = callback(draft, max_speed, accel_time, params.drawing.penup_mm/1000)
        self.trajectories = self.pad.fit_to_frame(trajectories)
        self.trajectories_computed = True

    def gui(self):
        params = self.sketch.params.params
        if imgui.button('Launch Swift'):
            self.itiba.launch_simulator()

        if imgui.button('Home'):
            self.msg_pub.publish('exec:Home')

        if imgui.button('Say hello'):
            self.send_msg('hello!!!')
        if imgui.button('Pick Tool'):
            self.msg_pub.publish('exec:Pick')
        imgui.same_line()
        if imgui.button('Release Tool'):
            self.msg_pub.publish('exec:Release')

        if imgui.button('Compute trajectories'):
            self.compute_trajectories()

        if imgui.button('Try dip'):
            self.dip_trajectory()

        if imgui.button('Print configuration'):
           tcp = self.robot.pen_tcp_pos(self.robot.q)
           q = self.robot.q
           print('TCP: ', list(tcp))
           print('q: ', list(q))

        if self.trajectories_computed and imgui.button('Plan trajectories'):
            cur_pos = self.robot.pen_tcp_pos(self.robot.q)
            trajectories = self.pad.augment_with_pen_orientation(self.trajectories, self.robot, self.robot.qr)
            # trajectories = [tu.rest_inout(traj, 1) for traj in trajectories]
            trajectories += [tu.trapezoidal_approach(trajectories[-1][-1], cur_pos)] # [tu.approach(cur_pos, trajectories[-1][-1])]
            self.itiba.plan_task_trajectories(trajectories)
            self.steps, self.fk_points = self.compute_steps(self.itiba.Qs)

        if self.itiba.Qs:
            if imgui.button('Simulate trajectory'):
                self.itiba.simulate()

            if state.has_ros and self.itiba.Qs:
                if imgui.button("Execute trajectory"):
                    imgui.open_popup('Execute')

                if imgui.begin_popup_modal('Execute').opened:
                    imgui.text('Are you sure?')
                    if imgui.button('OK'):
                        self.send_joint_trajectory(self.itiba.Qs, self.itiba.dQs)
                        imgui.close_current_popup()
                    imgui.same_line()
                    if imgui.button('Cancel'):
                        imgui.close_current_popup()
                    imgui.end_popup()


    def update(self):
        params = self.sketch.params.params
        # Udate drawing pad data
        self.pad.offset = params.drawing.plane_offset / 1000
        self.pad.pos_offset = [params.drawing.pos_offset_x, params.drawing.pos_offset_y]
        self.pad.update()

        if self.dirty or self.params_changed(*self.params_to_check):
            paths = self.pad.fit_to_box(self.draft, padding=params.drawing.padding, penup_mm=params.drawing.penup_mm)
            self.trajectories = self.pad.fit_to_frame(paths)
            self.trajectories_computed = False
            self.dirty = False

        #if state.has_ros:
        #    rospy.spin_once()

    def draw_gl(self):
        params = self.sketch.params.params
        width, height = self.sketch.width, self.sketch.height
        q = self.robot.q
        viewport = geom.make_rect(0, 0, width, height)
        v3d = self.v3d
        v3d.set_view(viewport, [params.tsm.xoffset, params.tsm.voffset, -params.tsm.dist],
                    geom.radians(np.array([params.tsm.rotx, params.tsm.roty, params.tsm.rotz])),
                    fov=40)

        v3d.color(1, 0.2)
        v3d.line_width(2.0)
        v3d.xy_grid(2, 21)

        v3d.frame(np.eye(4), alpha=0.4, scale=0.5)

        P, penP = self.robot.robot_polylines(q)
        v3d.color(0, 0.5, 1.0)
        v3d.line_width(8)
        v3d.polyline(P)

        v3d.line_width(2.0)
        penframe = self.robot.pen_frame(q)
        v3d.frame(penframe, alpha=0.9, scale=0.05)
        v3d.frame(self.robot.pen_tcp_frame(q), scale=0.02)

        v3d.line_width(9)
        v3d.color(1, 0.4)
        v3d.polyline(penP)
        v3d.line_width(1)

        self.pad.draw(v3d)

        if self.pad.is_valid():

            for i, trajectory in enumerate(self.trajectories):
                v3d.color(plut.default_color(i))
                self.draw_trajectory_3d(trajectory)

            if self.itiba.Qs:
                v3d.line_width(0.3)
                v3d.color(1.0, 0.1)
                v3d.stroke_shape(self.steps)
                v3d.color(1.0, 0.0, 0.0)
                v3d.points(self.fk_points)

        # Reset viewport
        v3d.rend.ctx.viewport = (0, 0, self.sketch.window_width, self.sketch.window_height)
