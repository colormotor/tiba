'''
o-O-o o-O-o o--o    O
  |     |   |   |  / \
  |     |   O--o  o---o
  |     |   |   | |   |
  o   o-O-o o--o  o   o

Painting robot utils.
© Daniel Berio (@colormotor) 2023 - ...

tiba_sketch - Tiba frontend interface for py5sketch app
Creates a UI for sending drawings to the robot and controlling it.
Requires https://github.com/colormotor/py5canvas to be installed
'''

import numpy as np
from importlib import reload
import importlib

from tiba import lqt
import imgui
from . import utils, svg, bezier
import roboticstoolbox as rtb
import pdb
import time
import threading

from . import interface
reload(interface)

from .interface import TibaInterface
from .robot import TibaRobot, tools
from . import matplotlyb as ply
from .config import cfg
from . import trajectory_utils as tu
from . import draw, geom, config, ui
from .gfx import renderer
from py5canvas import canvas

reload(renderer)
reload(tu)
reload(draw)
reload(ply)
reload(geom)
reload(ui)

from .draw import View3d, SketchView3d, GlView3d
from spatialmath import SE3, SO3, UnitQuaternion
from spatialmath.base import trnorm
import roboticstoolbox as rtb
from collections import namedtuple
import matplotlib.pyplot as plt
import swift
import matplotlib
import multiprocessing, sys


state = lambda: None
# state.has_ros = False

# rosloader = importlib.find_loader('rospy')
# # Load ROS if available, not strictly necessary if not using robot
# if rosloader is not None:
#     print("Loading ros")
#     import rospy
#     from std_msgs.msg import String
#     from trajectory_msgs.msg import JointTrajectory
#     from trajectory_msgs.msg import JointTrajectoryPoint
#     from sensor_msgs.msg import JointState
#     from std_msgs.msg import Float64MultiArray
#     from std_msgs.msg import Time
#     from std_msgs.msg import Header
#     from std_msgs.msg import Duration
#     state.has_ros = True

# # GUI helpers
# def imgui_title(text):
#     imgui.push_style_color(imgui.COLOR_HEADER, 0.3, 0.3, 0.3, 1.0)
#     imgui.selectable(text, True)
#     imgui.pop_style_color()

# def imgui_info_text(text):
#     imgui.push_style_color(imgui.COLOR_TEXT, 0.6, 0.6, 0.6, 1.0)
#     imgui.text(text)
#     imgui.pop_style_color()

# def imgui_text_separator(text):
#     imgui.push_style_color(imgui.COLOR_TEXT, 0.6, 0.6, 0.6, 1.0)
#     imgui.text(text)
#     imgui.same_line()
#     imgui.separator()
#     imgui.pop_style_color()

# def imgui_color_button(text, color):
#     if len(color) == 3:
#         color = np.concatenate([color, [1.0]])
#     imgui.push_style_color(imgui.COLOR_BUTTON, *color)
#     res = imgui.button(text)
#     imgui.pop_style_color()
#     return res

def plan_trapezoidal(samples,
                     max_speed,
                     accel_time,
                     penup_distance,
                     params):
    paths, velocities_, accelerations_ = tu.plan_polylines_time_optimal(samples,
                                                                  max_speed=max_speed,
                                                                  accel_time=accel_time,
                                                                        tol=params.config.topp_tol/1000)
    return paths, samples

def ema(data, alpha):
    ema_values = [data[0]]  # Initialize EMA with the first data point

    for i in range(1, len(data)):
        ema_values.append(alpha * data[i] + (1 - alpha) * ema_values[-1])

    return np.array(ema_values)

def plan_calligraphic(samples,
                      max_speed,
                      accel_time,
                      penup_distance,
                      params):
    simplified = [tu.simplify.dce(P, params.config.dce_thresh/1000) for P in samples]
    simplified = [P for P in simplified if len(P) > 1]
    if not simplified:
        return [], []

    paths = [np.hstack([P, np.zeros((len(P), 1))]) for P in simplified]
    up = np.array([0, 0, penup_distance])
    # Square pen ups
    # motor_plan = [paths[0][0] + up, paths[0]]
    # if len(paths) > 1:
    #     for i in range(1, len(paths)):
    #         P1 = paths[i-1]
    #         P2 = paths[i]
    #         if geom.distance(P1[-1], P2[0]) > 2e-3:
    #             motor_plan.append(P1[-1] + up)
    #             motor_plan.append(P2[0] + up)
    #         motor_plan.append(P2)
    # motor_plan.append(paths[-1][-1] + up)
    # Triangle pen ups
    motor_plan = [paths[0]]
    if len(paths) > 1:
        for i in range(1, len(paths)):
            P1 = paths[i-1]
            P2 = paths[i]
            if geom.distance(P1[-1], P2[0]) > 2e-3:
                motor_plan.append((P1[-1] + P2[0])/2 + up)
            motor_plan.append(P2)
    #motor_plan.append(paths[-1][-1] + up)

    motor_plan = np.vstack(motor_plan)
    # Add stop points
    motor_plan = np.vstack([motor_plan[0], motor_plan,motor_plan[-1]])
    trajectories = tu.plan_polylines_lqt([motor_plan],
                                         max_speed,
                                         accel_time=accel_time,
                                        isotropy=params.config.isotropy,
                                        sigma=params.config.smoothing,
                                        penups=True,
                                        order=params.config.order,
                                        opt_x0=False,
                                        subd=5,
                                        angle=params.config.angle)
    # Clip zs to zero
    print('Deviation from base: ', np.min(trajectories[0][:,2]))
    #trajectories[0][:,2] = ema(np.maximum(trajectories[0][:,2], 0.0), 0.1)
    return trajectories, simplified

def plan_min_jerk(samples,
                 max_speed,
                 accel_time,
                 penup_distance,
                 params):
    paths, simplified = tu.plan_polylines_jerk(samples,
                                                                params.config.dce_thresh/1000,
                                                                max_speed=max_speed,
                                                                max_accel_time=accel_time,
                                                                maxiter=params.config.max_iterations)
    return paths, simplified

def plan_preplanned(samples,
                    max_speed,
                 accel_time,
                 penup_distance,
                 params):
    return samples, samples

def plan_simplify_test(samples,
                 max_speed,
                 accel_time,
                 penup_distance,
                 params):
    simplified = [tu.simplify.dce(P, params.config.dce_thresh/1000) for P in samples]
    return samples, simplified


def plan_thick_resample(samples,
                        max_speed,
                        accel_time,
                        penup_distance,
                        params):
    chains = [np.array(chain[:,:2]) for chain in samples]
    widths = [np.array(chain[:,2]) for chain in samples]
    width_range_mm = [np.min(np.concatenate(widths))*1000, np.max(np.concatenate(widths))*1000]

    def map(x, input_range, output_range):
        return (output_range[0] + ((x - input_range[0])/(input_range[1] - input_range[0]))
                                    * (output_range[1] - output_range[0]))

    wrange = np.array(width_range_mm)/1000.0
    zrange = np.array([0.0, -params.config.max_depth])/1000.0

    def width_to_z(w):
        return map(w, wrange, zrange)

    def z_to_width(z):
        return map(z, zrange, wrange)

    chains_3d = [np.hstack([Cp, width_to_z(w.reshape(-1,1))]) for Cp, w in zip(chains, widths)]

    if params.config.resample_topp:
        trajectories,_,_ = tu.plan_bezier_chains_time_optimal(chains_3d, cfg.cartesian_speed, cfg.acceleration_time)
    elif params.config.resample_jerk:
        trajectories = tu.plan_bezier_chains_jerk(chains_3d, cfg.cartesian_speed, cfg.acceleration_time, maxiter=1000, bezier_subd=3, method='NLOPT_PRAXIS') #, bezier_tol=0.05)
    else:
        trajectories = [tu.resample_bezier(chain, max_speed, accel_time) for chain in chains_3d]
    return trajectories, [P[:,:2] for P in samples]
    
def plan_polylines(planning_callback,
                   samples,
                   max_speed,
                   accel_time,
                   penup_distance,
                   trajectory_includes_penups=False,
                   params={},
                   debug=True):
    ''' General function to plan a 2d trajectory so it is projected to 3d on the drawing pad
    Does not handle orientation, only position
    '''
    config.progress_text = 'Planning trajectory'
    config.progress = 0
    config.progress_max = len(samples)
    if not trajectory_includes_penups:
        # Plan each trajectory in 2d and then concatenate lifting motions
        if samples[0].shape[1] < 3:
            samples = [np.hstack([P, np.zeros((len(P), 1))]) for P in samples]
        # up = np.array([0, 0, penup_distance])
        # samples = ([np.array([samples[0][0]+up, samples[0][0]])]
        #                     + samples
        #                     + [np.array([samples[-1][-1], samples[-1][-1]+up])])
        paths, simplified = planning_callback(samples,
                                              penup_distance=penup_distance,
                                              max_speed=max_speed,
                                              accel_time=accel_time,
                                              params=params)
        paths = [P for P in paths if len(P) > 1]

        print('lens', [len(P) for P in paths])
        if not paths:
            return []
        trajectories = [paths[0]]
        if len(paths) > 1:
            for i in range(len(paths)-1):
                P1 = paths[i]
                P2 = paths[i+1]
                if geom.distance(P1[-1], P2[0]) > 2e-3:
                    trajectories.append(tu.lifting_motion(P1[-1], P2[0], penup_distance, max_accel_time=accel_time))
                trajectories.append(P2)

    else:
        # Create one motor plan with pen-ups and plan single trajectory
        trajectories, simplified = planning_callback(samples,
                                                     penup_distance=penup_distance,
                                                     max_speed=max_speed,
                                                     accel_time=accel_time,
                                                     params=params)


    velocities = [np.diff(P, axis=0) / cfg.dt for P in trajectories]
    accelerations = [np.diff(P, n=2, axis=0) / (cfg.dt**2) for P in trajectories]
    max_accel = max_speed / accel_time

    if debug:
        ply.figure(title='Cartesian trajectory')

        Tv = 0.0
        Ta = 0.0

        for i, X in enumerate(trajectories):
            #speed = velocities[i] #tu.compute_speed(X, cfg.dt)
            #accel = accelerations[i] #tu.compute_accel(X, cfg.dt)
            tv = np.linspace(0, (len(velocities[i])-1)*cfg.dt, len(velocities[i])) + Tv
            ta = np.linspace(0, (len(accelerations[i])-1)*cfg.dt, len(accelerations[i])) + Ta
            ply.subplot(1, 1, title='Velocity')
            ply.plot(tv, velocities[i][:,0], color=ply.categorical_color(i))
            ply.plot(tv, velocities[i][:,1], color=ply.categorical_color(i))
            ply.plot(tv, velocities[i][:,2], color='b', alpha=0.5)
            ply.subplot(1, 2, title='Acceleration')
            ply.plot(ta, accelerations[i][:,0], color=ply.categorical_color(i))
            ply.plot(ta, accelerations[i][:,1], color=ply.categorical_color(i))
            ply.plot(ta, accelerations[i][:,2], color='b', alpha=0.5)
            Tv = tv[-1] + cfg.dt
            Ta = ta[-1] + cfg.dt

        ply.subplot(1, 1, title='Velocity')
        ply.rectangle([0, -max_speed], [Tv, max_speed], color='k', alpha=0.1, fill=True)
        ply.subplot(1, 2, title='Acceleration')
        ply.rectangle([0, -max_accel], [Ta, max_accel], color='k', alpha=0.1, fill=True)


        ply.subplot(2, 1, title='Trajectory 2d')
        ply.stroke(samples, 'k', alpha=0.25)
        ply.stroke(simplified, 'r', ls=':', lw=1)
        ply.stroke(trajectories, 'k')
        ply.axis_equal(True)

        ply.subplot(2, 2, title='Trajectory 3d')

        box = geom.bounding_box(trajectories)

        for X in trajectories:
            X = np.array(X)
            vals = (X[:,2])/(box[1][2] - box[0][2])
            ply.stroke3d(X, color=[ply.cmap(z, 'magma') if z >=0 else ply.cmap(-z, 'viridis') for z in vals])
        ply.show()
    return trajectories

dip_pos = np.array([0.11812825731443674, -0.407704314564537, 0.10439764358100054,
                    -0.008107937414934388, 0.8524478856525898, -0.5214394134973193, -0.036983802520516196])

#  [-0.7743211365357173, -0.10300303344311892, -0.50291393240498, -2.5627437008673333, -0.1284246296894626, 2.401976760003019, 0.7079839864811134]
#

planner_names = ['Trapezoidal', 'Minimum Jerk', 'Calligraphic', 'Test simplify', 'Thick resample', 'Preplanned']

class TibaSketch:
    def __init__(self, flags=['drawing']):
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

        self.flags = flags
        self.arcball = geom.ArcBall()
        self.thread_funcs = []
        self.itiba = None

    def init(self, sketch,
             callback=None,
             input_tools=None,
             panda=None,
             gripper=None,
             init_node=''):

        # The py5canvas "sketch" that will be used to run this
        self.sketch = sketch
        self.trajectory_callback = callback

        # Actual interface to the robot. Here this is not used to control the robot directly
        # but to plan and execute trajectories
        if input_tools is None:
            self.tools = dict(tools)
        else:
            self.tools = input_tools

        # Actual interface to robot (computes IK, holds reference to robot model and drawing pad
        print("Init interface")

        self.itiba = TibaInterface(panda=panda, gripper=gripper, init_node=init_node)
        # 3d view for the robot
        self.v3d = GlView3d(sketch)
        # Reference to the drawing pad
        self.pad = self.itiba.pad

        # Visualization limits, trajectories will be quantized to fit these numbers
        self.max_steps = 500
        self.max_trajectory_points = 200

        # Setup robot
        tool_data = self.get_tool_by_index(self.params.config.tool)
        self.set_tool(tool_data)

        # For launching interface scripts
        self.cur_script = -1

        # self.robot.q = self.robot.qr
        # self.robot.q = np.array([-0.7743211365357173, -0.10300303344311892, -0.50291393240498, -2.5627437008673333, -0.1284246296894626, 2.401976760003019, 0.7079839864811134])

        self.draft = []
        self.draft_fitted = []
        self.clear()

        # print("#################################################")
        # print("STARTING UP ROS, IF THIS HANGS CTRL-C TO CONTINUE")
        # if state.has_ros:
        #     try:
        #         self.msg_pub = rospy.Publisher('/tiba/msg', String, queue_size=10)
        #         self.jtraj_pub = rospy.Publisher('/tiba/jtraj', JointTrajectory, queue_size=10)
        #         rospy.Subscriber('/tiba/response', String, self.response_callback)
        #         rospy.Subscriber('/tiba/q', JointState, self.state_callback)
        #         rospy.init_node('motion_generator', log_level=rospy.ERROR, anonymous=True)
        #         self.send_msg('start: Tiba sketch connecting')
        #     except rospy.exceptions.ROSException as e:
        #         print("Ros issue")
        #         state.has_ros = False

        self.threads = []

    def release(self):
        self.v3d.release()

    # def send_msg(self, data):
    #     print("Sending msg", data)
    #     self.msg_pub.publish(data)

    # def response_callback(self, data):
    #     print("Received response: ", data.data)

    # def state_callback(self, data):
    #     #print("Received state")
    #     self.robot.q = np.array(data.position)
    #     #self.itiba.Qs = []
    #     #self.steps, self.fk_points = [], []
    #     #self.trajectories_computed = False
    #     #self.trajectories = []

    #@staticmethod
    def add_parameters(self, params_to_update):
        ''' Add parameters to the sketch '''
        params = {}
        params.update({
            'Config': {
                'tool': (0, {'hide': True}),
                'Cart planner': (0, {'selection':planner_names}),
                'DCE thresh' : (250, {'min':0.0, 'max':500, 'hide':True}),
                'Max iterations' : (450, {'min':50, 'max':1000, 'hide':True}),
                'Topp tol': (100.0, {'min':0.0, 'max':100.0, 'hide':True}),
                'Isotropy': (1.0, {'min':0.0, 'max':1.0, 'hide':True}),
                'Angle': (50.0, {'min':-90.0, 'max':90.0, 'hide':True}),
                'Smoothing': (0.0, {'min':0.01, 'max':15.0, 'hide':True}),
                'Order': (3, {'min':2, 'max':5, 'hide':True}),
                'Max depth': (4.0, {'min':0.0, 'max':7.0, 'hide':True}),
                'Resample TOPP': False,
                'Resample Jerk': False,
            }
        })

        if 'drawing' in self.flags:
            params.update({'Drawing':  {
              'Plane offset': (0.0, {'min':-20,'max': 20}),
              'Padding': (80, {'min':0.0, 'max':850}),
              'Penup mm': (50, {'min':0.0, 'max':200}),
              'Pos offset x': (0.0, {'min': -1, 'max': 1}),
              'Pos offset y': (0.0, {'min': -1, 'max': 1}),
            }
                           })

        params.update({
            'Tsm': {
              'Dist': (4.0, {'min':0, 'max':10}),
              'Xoffset': (0.0, {'min':-1, 'max':1}),
              'Voffset': (0.0, {'min':-1, 'max':1}),
              'rotx': (275, {'min':0, 'max':360}),
              'roty': (0, {'min':0, 'max':360}),
              'rotz': (138, {'min':0, 'max':360}),
            }
        })
        params_to_update.update(params)

    @property
    def c(self):
        return self.sketch.canvas

    @property
    def robot(self):
        return self.itiba.robot

    @property
    def params(self):
        return self.sketch.params.params

    def set_drawing_data(self, data):
        self.drawing_data = data

    def get_tool_names(self):
        names = list(self.tools.keys())
        return names

    def get_tool_name(self, index):
        names = list(self.tools.keys())
        return names[index]

    def get_tool_by_index(self, index):
        names = list(self.tools.keys())
        return self.tools[names[index]]

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

    # def send_joint_trajectory(self, Qs, dQs):
    #     jtraj = JointTrajectory()
    #     jtraj.header = Header()
    #     jtraj.header.stamp = rospy.Time.now()
    #     jtraj.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6',
    #                             'panda_joint7']
    #     for i, (q, dq) in enumerate(zip(Qs, dQs)):
    #         p = JointTrajectoryPoint()
    #         p.positions = q
    #         p.velocities = dq
    #         #p.time_from_start = Duration(0, int(1e+9*cfg.robot_dt*i))
    #         jtraj.points.append(p)
    #     self.jtraj_pub.publish(jtraj)
    #     rospy.loginfo("Sent joint trajectory")

    def params_changed(self, *args):
        for key in args:
            if key in self.sketch.gui.changed:
                return True
        return False

    def clear(self):
        self.steps = []
        self.fk_points = []
        self.trajectories = []
        self.trajectories_2d = []
        self.draft_fitted = []
        self.dirty = True
        self.trajectories_computed = False
        self.itiba.clear()

    def set_tool(self, tool_dict):
        self.clear()
        self.itiba.set_tool(tool_dict)

    def set_draft(self, draft):
        self.clear()
        self.draft = draft

    def plan_joint_approach(self, q):
        self.itiba.plan_joint_approach(q)
        self.steps, self.fk_points = self.compute_steps(self.itiba.Qs)
        self.trajectories_computed = True

    def plan_script(self, path, **kwargs):
        self.itiba.plan_script(path, **kwargs)
        self.steps, self.fk_points = self.compute_steps(self.itiba.Qs)
        self.trajectories_computed = True

    def plan_task_trajectories(self, trajectories, tip_frame='tiba_pen_tcp'):
        self.itiba.plan_task_trajectories(trajectories, tip_frame=tip_frame)
        self.steps, self.fk_points = self.compute_steps(self.itiba.Qs)
        self.trajectories_computed = True


    def dip_trajectory(self):
        up = np.array([0, 0, 1, 0, 0, 0, 0])
        start_offset = 70/1000
        dip_offset = 35/1000
        clean_offset = 10/1000
        dip_start = dip_pos + up*start_offset

        print(dip_start)
        dip_start = tu.align_pos_z(dip_start, np.array([0, 0, -1]))
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

        self.plan_task_trajectories(trajectories)
        # self.itiba.plan_task_trajectories(trajectories)
        # self.steps, self.fk_points = self.compute_steps(self.itiba.Qs)
        # self.trajectories_computed = True
        # print('Dip trajectory')

    def get_fitted_draft_and_trajectories(self):
        if self.trajectories_computed:
            return self.draft_fitted, self.trajectories_2d
        else:
            return self.draft, None

    def compute_trajectories(self, debug=False):
        params = self.params
        if not self.draft:
            print("No draft")
            return

        max_speed = cfg.cartesian_speed
        accel_time = cfg.acceleration_time
        traj_callback = self.trajectory_callback
        trajectory_includes_penup = False

        if traj_callback is None:
            # Select planner from parameters
            if planner_names[params.config.cart_planner] == 'Trapezoidal':
                traj_callback = plan_trapezoidal
            elif planner_names[params.config.cart_planner] == 'Minimum Jerk':
                traj_callback = plan_min_jerk
            elif planner_names[params.config.cart_planner] == 'Calligraphic':
                traj_callback = plan_calligraphic
                trajectory_includes_penup = True
                max_speed *= 2
            elif planner_names[params.config.cart_planner] == 'Test simplify':
                traj_callback = plan_simplify_test
            elif planner_names[params.config.cart_planner] == 'Thick resample':
                traj_callback = plan_thick_resample
            elif planner_names[params.config.cart_planner] == 'Preplanned':
                traj_callback = plan_preplanned

        if traj_callback is None:
            raise ValueError('No trajectory callback')

        time.sleep(0.01)
        draft = self.pad.fit_to_box(self.draft, padding=params.drawing.padding, penup_mm=0.0)

        #pdb.set_trace()
        trajectories = plan_polylines(traj_callback, draft, max_speed, accel_time,
                                      params.drawing.penup_mm/1000,
                                      trajectory_includes_penup,
                                      params,
                                      debug=debug)

        def adapt_xy(P):
            if type(P)==list:
                return [adapt_xy(Q) for Q in P]
            return np.vstack([-P[:,1], -P[:,0]]).T

        self.trajectories_2d = adapt_xy([np.array(P)[:,:2] for P in trajectories])
        self.draft_fitted = adapt_xy([np.array(P)[:,:2] for P in draft])

        self.trajectories = self.pad.fit_to_frame(trajectories)
        self.trajectories_computed = True

    def script_gui(self):
        itiba = self.itiba
        selected_script = {}
        if itiba.scripts:
            names = list(itiba.scripts.keys())
            clicked, self.cur_script = imgui.listbox("Scripts", self.cur_script, names)
            if imgui.button('Reload scripts'):
                itiba.reload_scripts()

            if -1 < self.cur_script < len(names):
                name = names[self.cur_script]
                script = itiba.scripts[name]

                imgui.same_line()
                if ui.imgui_color_button('Load script'):
                    selected_script = script

                if 'desc' in script:
                    ui.imgui_info_text('Description: ' + script['desc'])
                if 'args' in script:
                    ui.imgui_info_text('Arguments: ')
                    for key, val in script['args'].items():
                        imgui.text(f'{key}: {val}')

        return selected_script

    def gui(self):
        params = self.params
        if imgui.button('Launch Swift'):
            self.itiba.launch_simulator()
        imgui.same_line()
        if imgui.button('Say hello'):
            self.itiba.send_msg('hello!!!')
        imgui.same_line()
        if imgui.button('Print pose'):
           tcp = self.robot.pen_tcp_pos(self.robot.q)
           q = self.robot.q
           print('TCP: ', list(tcp))
           print('q: ', list(q))
           hand = self.robot.get_pose(None, 'tiba_hand_tcp')
           print('Hand TCP pos', list(hand.t))
        button_color = [1.0, 0.5, 0.0, 1.0]
        if imgui.button('Update aruco'):
            self.itiba.go_aruco()
            
        _, cfg.speed_factor = imgui.slider_float('Speed factor', cfg.speed_factor, 0.1, 1.0)

        if 'drawing' in self.flags:
            ui.imgui_text_separator("Scripts")
            script = self.script_gui()
            if script:
                args = script['args'] if 'args' in script else {}
                self.plan_script(script['file'], **args)
                #if self.itiba.Qs:
                #    self.itiba.plot_trajectory_stats()

            ui.imgui_text_separator('Robot Actions')
            if imgui.button('Home'):
                self.itiba.send_msg('goto:Home')
                #self.msg_pub.publish('goto:Home')
            # imgui.same_line()
            # if imgui.button('Pick Tool'):
            #     self.msg_pub.publish('exec:Pick')
            # imgui.same_line()
            # if imgui.button('Release Tool'):
            #     self.msg_pub.publish('exec:Release')
            # if imgui.button('Try dip'):
            #     self.dip_trajectory()

            ui.imgui_text_separator('Trajectory')
            changed, self.itiba.safe_mode = ui.imgui_color_checkbox("Safe mode", self.itiba.safe_mode)

            if imgui.button('Clear plans'):
                self.clear()

            if imgui.button('Compute trajectories only'):
                # self.threads.append(threading.Thread(target=lambda: self.compute_trajectories(debug=True),
                #                                     name="Planning trajectories"))
                self.compute_trajectories(debug=True)

            must_plan = False

            if ui.imgui_color_button('Compute and plan'):
                # self.compute_trajectories()
                # must_plan = True
                self.threads.append(threading.Thread(target=self.compute_trajectories,
                                                    name="Planning trajectories"))
                def plan_ik_thread():
                    time.sleep(0.2)
                    cur_pos = self.robot.pen_tcp_pos(self.robot.q)
                    trajectories = self.pad.augment_with_pen_orientation(self.trajectories, self.robot, self.robot.qr)
                    #trajectories = [tu.rest_inout(traj, 3) for traj in trajectories]
                    trajectories += [tu.trapezoidal_approach(trajectories[-1][-1], cur_pos)] # [tu.approach(cur_pos, trajectories[-1][-1])]
                    self.itiba.plan_task_trajectories(trajectories)
                    self.task_trajectories_augmented = trajectories

                def compute_polygons_and_plot():
                    steps, fk_points = self.compute_steps(self.itiba.Qs)
                    self.steps, self.fk_points = steps, fk_points
                    #if self.itiba.Qs:
                    #    self.itiba.plot_trajectory_stats()
                    # self.itiba.plot_trajectory_stats(self.task_trajectories_augmented)

                    #self.steps, self.fk_points = self.compute_steps(self.itiba.Qs)

                self.threads.append(threading.Thread(target=plan_ik_thread, name="Planning IK"))
                self.threads.append(compute_polygons_and_plot)

            # if self.trajectories_computed and must_plan: #(must_plan or imgui.button('Plan trajectories')):

            #     cur_pos = self.robot.pen_tcp_pos(self.robot.q)
            #     trajectories = self.pad.augment_with_pen_orientation(self.trajectories, self.robot, self.robot.qr)
            #     # trajectories = [tu.rest_inout(traj, 1) for traj in trajectories]
            #     trajectories += [tu.trapezoidal_approach(trajectories[-1][-1], cur_pos)] # [tu.approach(cur_pos, trajectories[-1][-1])]
            #     self.itiba.plan_task_trajectories(trajectories)
            #     self.steps, self.fk_points = self.compute_steps(self.itiba.Qs)

            if self.itiba.Qs: #motions: #Qs:
                imgui.separator()
                if not self.itiba.has_ros:
                    if ui.imgui_color_button('Simulate trajectory'):
                        #self.threads.append(threading.Thread(target=self.itiba.simulate, name='nogui'))
                        self.itiba.simulate()
                        self.robot.q = tsketch.itiba.Qs[-1]
                else:
                    if imgui.button('Simulate trajectory'):
                        self.itiba.simulate()

                    if ui.imgui_color_button("Execute trajectory"):
                        # If safe mode is active or we detected a problem ask twice
                        if self.itiba.safe_mode or not self.itiba.trajectory_status:
                            imgui.open_popup('Execute')
                        else:
                        # Othwerwise just send it
                            if 'no_server' in self.flags:
                                self.itiba.execute()
                            else:
                                self.itiba.send_joint_trajectory()
                                #
                    if imgui.begin_popup_modal('Execute').opened:
                        imgui.text('Are you sure?')
                        if imgui.button('OK'):
                            if 'no_server' in self.flags:
                                self.itiba.execute()
                            else:
                                self.itiba.send_joint_trajectory()
                            imgui.close_current_popup()
                        imgui.same_line()
                        if imgui.button('Cancel'):
                            imgui.close_current_popup()
                        imgui.end_popup()

        imgui.separator()
        # ui.imgui_info_text('Drag to rotate')
        # ui.imgui_info_text('Ctrl+drag to zoom, Shift+drag to pan')


        # Threading
        if len(self.threads):
            if type(self.threads[0]) == threading.Thread:
                if self.threads[0].ident is None:
                    config.state.progress_text = self.threads[0].name
                    config.state.progress = 0
                    config.state.progress_active = True
                    if self.threads[0].name != 'nogui':
                        imgui.open_popup("Progress")
                    self.threads[0].start()
                elif not self.threads[0].is_alive():
                    self.threads[0].join()
                    self.threads.pop(0)
                    config.state.progress_active = False
            else:
                config.state.progress_active = False
                self.threads[0]()
                self.threads.pop(0)

        imgui.set_next_window_size(200, 70)
        if imgui.begin_popup_modal("Progress").opened:
            imgui.text(config.state.progress_text)
            imgui.progress_bar(config.state.progress / config.state.progress_max)

            if not config.state.progress_active:
                imgui.close_current_popup()
            imgui.end_popup()


    def hide_config_except(self, *args):
        dont_hide = ['tool', 'cart_planner'] + list(args)
        for key, val in self.sketch.params.gui_params['Config'].items():
            if 'hide' not in val[1]:
                continue
            if key not in dont_hide:
                self.sketch.params.gui_params['Config'][key][1]['hide'] = True
            else:
                self.sketch.params.gui_params['Config'][key][1]['hide'] = False

    def update(self):
        if 'config.cart_planner' in self.sketch.gui.changed:
            self.clear()
        if planner_names[self.params.config.cart_planner] == 'Trapezoidal':
            self.hide_config_except('Topp tol')
        elif planner_names[self.params.config.cart_planner] == 'Minimum Jerk':
            self.hide_config_except('DCE thresh', 'Max iterations')
        elif planner_names[self.params.config.cart_planner] == 'Calligraphic':
            self.hide_config_except('DCE thresh', 'Isotropy', 'Angle', 'Smoothing', 'Order')
        elif planner_names[self.params.config.cart_planner] == 'Thick resample':
            self.hide_config_except('Max depth', 'Resample TOPP', 'Resample Jerk')
        else:
            self.hide_config_except()

        # Update arcball
        self.arcball.update(self.sketch.width,
                            self.sketch.height)

        params = self.params 
        # Udate drawing pad data
        if 'drawing' in self.flags:
            self.pad.offset = params.drawing.plane_offset / 1000 + self.itiba.pen_offset_base_mm / 1000
            self.pad.pos_offset = [params.drawing.pos_offset_x, params.drawing.pos_offset_y]
            self.pad.update()

            if self.dirty or self.params_changed(*self.params_to_check):
                paths = self.pad.fit_to_box(self.draft, padding=params.drawing.padding, penup_mm=params.drawing.penup_mm)
                self.trajectories = self.pad.fit_to_frame(paths)
                self.trajectories_computed = False
                self.dirty = False


        # Update handeye info
        self.itiba.update_handeye()
        
    def draw(self, drawing_3d=[]):
        c = self.sketch.canvas
        c.text_size(20)
        c.fill(1)
        c.text([10, 30], 'Current tool: ' + self.get_tool_name(self.params.config.tool))
        c.text_size(12)
        if self.itiba.trajectory_status:
            c.fill(0,1,0)
            c.text([10, 45], 'Trajectory status: OK')
        else:
            c.fill(1,0,0)
            c.text([10, 45], 'Trajectory status: Error')

        c.text_size(12)
        c.fill(0.5)
        c.text([c.width-290, 20], 'Drag to rotate, Ctrl-drag to zoom, Shift-drag to zoom')

        handeye = self.itiba.handeye
        if handeye is not None and handeye.cam_image is not None:
            cam_image = np.array(handeye.cam_image)
            w = cam_image.shape[1]
            h = cam_image.shape[0]
            if self.itiba.handeye_valid():
                size = 50/1000
                points = [[0,0,0],
                        [size,0,0],
                        [0,size,0],
                        [0,0,size]]
                projected_points = handeye.project_points(points, 'aruco', fudge=False)
                cc = canvas.Canvas(w, h)
                cc.image(cam_image)
                cc.no_fill()
                cc.stroke_weight(10)
                cc.stroke(255, 0, 0)
                #c.circle(200, 200, 100)
                cc.line(projected_points[0], projected_points[1])
                cc.stroke(0, 255, 0)
                cc.line(projected_points[0], projected_points[2])
                cc.stroke(0, 0, 255)
                cc.line(projected_points[0], projected_points[3])

                if self.pad.is_valid():
                    points = self.pad.get_box_points()
                    projected_points = handeye.project_points(points, 'world')
                    cc.polygon(projected_points)

                    for i, trajectory in enumerate(self.trajectories):
                        projected_points = handeye.project_points(trajectory, 'world')
                        cc.stroke(0)
                        cc.stroke_weight(3.0)
                        cc.polyline(projected_points)
                        
                cam_image = cc.get_image()
            else:
                pass #print("Invalid handeye")
            ref_h = 180
            c.image(cam_image, 0, c.height-ref_h, (w/h)*ref_h, ref_h)

    def begin_draw_gl(self):
        params = self.params
        width, height = self.sketch.width, self.sketch.height
        viewport = geom.make_rect(0, 0, width, height)
        v3d = self.v3d

        if self.sketch.dragging:
            rot = self.arcball.affine
        else:
            xyz = [params.tsm.rotx,
                   params.tsm.roty,
                   params.tsm.rotz]
            # NB need to use consistent rotations for the sync to work
            # rot_euler_xyz and arcball both use scipy Rotation as a backend
            rot = geom.rot_euler_xyz(xyz, degrees=True)

        pos = [params.tsm.xoffset, params.tsm.voffset, -params.tsm.dist]
        v3d.set_view(viewport, [0, 0, 0],
                     geom.trans_3d(pos)@
                     rot@geom.rotx_3d(geom.radians(90)),
                    fov=40)

    def end_draw_gl(self):
        # Reset viewport
        self.v3d.rend.ctx.viewport = (0, 0, self.sketch.window_width, self.sketch.window_height)

    def draw_gl(self):
        self.begin_draw_gl()
        self.draw_gl_main()
        self.end_draw_gl()

    def draw_gl_main(self):
        q = self.robot.q
        v3d = self.v3d

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

        for motion in self.itiba.motions:
            v3d.frame(self.robot.pen_tcp_frame(motion.Qs[-1]), scale=0.1)

        cur_q = self.itiba.editor.get_cur_position()
        if cur_q:
            P, penP = self.robot.robot_polylines(cur_q)
            v3d.color(0.5, 0.0, 0.0, 0.5)
            v3d.line_width(2)
            v3d.polyline(P)
            penframe = self.robot.pen_frame(cur_q)
            v3d.frame(penframe, alpha=0.3, scale=0.05)

        v3d.line_width(9)
        v3d.color(1, 0.4)
        v3d.polyline(penP)
        v3d.line_width(1)

        self.pad.draw(v3d)

        if self.pad.is_valid():
            for i, trajectory in enumerate(self.trajectories):
                v3d.color(ply.default_color(i))
                self.draw_trajectory_3d(trajectory)

            if self.itiba.motions: #Qs:
                v3d.line_width(0.3)
                v3d.color(1.0, 0.1)
                if self.steps:
                    v3d.stroke_shape(self.steps)
                v3d.color(1.0, 0.0, 0.0)
                if len(self.fk_points):
                    v3d.points(self.fk_points)

        if self.itiba.handeye_valid():
            self.itiba.handeye.draw_gl(self.itiba, v3d)
    

    def mouse_pressed(self, button, modifiers):
        params = self.params
        xyz = [params.tsm.rotx,
               params.tsm.roty,
               params.tsm.rotz]
        self.arcball.from_euler(xyz,
                           degrees=True)
        self.arcball.click(self.sketch.mouse_pos)

    def mouse_dragged(self, button, modifiers):
        params = self.params
        inc = 0.1
        if modifiers & 2: # ctrl (zoom)
            params.tsm.dist += self.sketch.mouse_delta[1]*inc*0.5
        elif modifiers & 1: # Shift (pan)
            params.tsm.xoffset += self.sketch.mouse_delta[0]*inc*0.1
            params.tsm.voffset -= self.sketch.mouse_delta[1]*inc*0.1
        else:
            self.arcball.drag(True, self.sketch.mouse_pos)
            # Copy back to params
            rot = self.arcball.euler_xyz(degrees=True)
            (params.tsm.rotx,
             params.tsm.roty,
             params.tsm.rotz) = rot
