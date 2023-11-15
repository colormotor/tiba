#!/usr/bin/env python3
import numpy as np
import importlib
panda_loader = importlib.find_loader('panda_py')
if panda_loader is not None:
    import panda_py
    from panda_py import libfranka
    from panda_py import controllers
    has_robot = True
else:
    has_robot = False

from motor import lqt
import imgui
from polygonsoup import utils, geom, svg
import roboticstoolbox as rtb
import tiba

import time
from importlib import reload
reload(tiba)
reload(tiba.draw)
reload(tiba.robot)
from tiba.interface import FrankaButton, TibaInterface
from tiba.robot import TibaRobot
from tiba.draw import View3d, SketchView3d, GlView3d
from tiba import planner, cfg
import tiba.trajectory_utils as tu
import tiba.gfx.renderer as renderer
reload(renderer)
reload(tu)
from spatialmath import SE3, SO3, UnitQuaternion
from spatialmath.base import trnorm
import roboticstoolbox as rtb
reload(planner)
from collections import namedtuple
import swift
import matplotlib

rosloader = importlib.find_loader('rospy')
if rosloader is not None:
    import rospy
    from std_msgs.msg import String
    from trajectory_msgs.msg import JointTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint
    from std_msgs.msg import Float64MultiArray
    from std_msgs.msg import Time
    from std_msgs.msg import Header
    from std_msgs.msg import Duration
    has_ros = True

    msg_pub = rospy.Publisher('/tiba/msg', String, queue_size=10)
    jtraj_pub = rospy.Publisher('/tiba/jtraj', JointTrajectory, queue_size=10)
    rospy.init_node('motion_editor', anonymous=True)
else:
    has_ros = False

np.set_printoptions(suppress=True)

params = {'Seed': (5, {'min': 0, 'max': 100}),
          'Gripper mm': (34.0, {'min': 2, 'max': 200}),
          'Gripper state': False,
          'Guiding': {
              'x': (True, {'sameline':0}),
              'y': (True, {'sameline':1}),
              'z': (True, {'sameline':1}),
              'rot': (True, {'sameline':0}),
              'elbow': (True, {'sameline':1})
          },
          'Traj': {
              'Sigma': (0.5, {'min': 0.1, 'max':3.0}),
              'Isotropy': (0.5, {'min': 0.1, 'max':1.0}),
              'Angle': (0.0, {'min': -90, 'max':90}),

              },
          'Drawing': {
              'Plane offset' : (0.0, {'min':-20,'max': 20}),
              'Padding' : (80, {'min':0.0, 'max':450}),
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


def random_q():
    return list(np.random.uniform(-np.pi, np.pi, 7))


def on_event(event):
    print(event)
    if circle_button(event):
        state.grasped = not state.grasped
        if state.grasped:
            print('Grasp')
            grasp()
        else:
            print('Ungrasp')
            gripper.stop()
            gripper.move(100.0/1000, 0.2)
    if cross_button(event):
        if state.plane_calibration:
            p = robot.pen_tcp_pos(robot.q)[:3]
            pad.add_point(p)
        else:
            editor.add_joint_step(robot.q)
    if check_button(event):
        if state.grasped:
            editor.add_grip_step(params.gripper_mm)
        else:
            editor.add_ungrip_step()

def grasp():
    print('grasping ', params.gripper_mm)
    gripper.grasp(params.gripper_mm/1000, 0.2, 10, 0.04, 0.04)


state = lambda: None
state.steps = []
state.fk_points = []
state.plane_calibration = False
state.grasped = False
state.trajectories = []
state.trajectories_2d = []
state.tcp_pos = np.zeros(7)

circle_button = FrankaButton('circle')
cross_button = FrankaButton('cross')
check_button = FrankaButton('check')

if has_robot:
    hostname = '172.16.0.2'
    panda = panda_py.Panda(hostname)
    desk = panda_py.Desk(hostname, 'colormotor', 'Rapidograph22!')
    gripper = libfranka.Gripper(hostname)
    desk.listen(on_event)
else:
    gripper = None
    desk = None
    panda = None

robot = TibaRobot(45, 11.5) # Molotow
# robot = TibaRobot(60, 0, 0)
itiba = TibaInterface(robot, panda, gripper)
editor = itiba.editor
pad = itiba.pad
robot.q = robot.qr


v3d = None
glyphs = []

def compute_steps(robot, Qs, max_steps=500):
    n = min(len(Qs), max_steps)
    I = np.linspace(0, len(Qs)-1, n, dtype=int)
    polylines = []
    points = []

    for i in I:
        q = Qs[i]
        P, penP = robot.robot_polylines(q)
        polylines.append(P)
        polylines.append(penP)
        points.append(robot.pen_tcp_pos(q)[:3])
    return polylines, points

def draw_steps(robot, v3d, Qs, max_steps=150):
    n = min(len(Qs), max_steps)
    I = np.linspace(0, len(Qs)-1, n, dtype=int)
    v3d.color(1, 0.25)
    v3d.line_width(1.0)
    pts = []
    for i in I:
        q = Qs[i]
        P, penP = robot.robot_polylines(q)
        v3d.polyline(P)
        v3d.polyline(penP)
        pts.append(robot.pen_tcp_pos(q)[:3])

    v3d.color(1, 0, 0, 0.4)
    v3d.line_width(4)
    v3d.polyline(pts)

def draw_trajectory_2d(P, max_steps=200):
    n = min(len(P), max_steps)
    I = np.linspace(0, len(P)-1, n, dtype=int)
    P = np.array(P)[I]
    polyline(P)

def draw_trajectory_3d(v3d, P, max_steps=200):
    n = min(len(P), max_steps)
    I = np.linspace(0, len(P)-1, n, dtype=int)
    P = np.array(P)[I]
    v3d.polyline(P)


def setup():
    sketch.set_gui_callback(gui)
    create_canvas(700, 600) #, 300)
    set_color_scale(1.0)

    global params, v3d
    params = sketch.parameters(params)
    v3d = GlView3d(sketch) #SketchView3d(sketch)

    global glyphs
    glyphs = svg.load_svg_polylines('ens.svg')
    destrect = geom.make_rect(0, height-300, 400, 300)
    glyphs = geom.transform_to_rect(glyphs, destrect)


def exit():
    print("Exit")
    if has_robot:
        desk.stop_listen()


def plan_trajectories(trajectories, rest=0.05): #15):
    itiba.plan_trajectories(trajectories, rest)
    state.steps, state.fk_points = compute_steps(robot, state.Qs)

def send_joint_trajectory(Qs, dQs):
    jtraj = JointTrajectory()
    jtraj.header = Header()
    jtraj.header.stamp = rospy.Time.now()
    jtraj.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6',
                              'panda_joint7']
    for i, (q, dq) in enumerate(zip(Qs[:100], dQs)):
        p = JointTrajectoryPoint()
        p.positions = q
        p.velocities = dq
        #p.time_from_start = Duration(0, int(1e+9*cfg.robot_dt*i))
        jtraj.points.append(p)
    jtraj_pub.publish(jtraj)
    rospy.loginfo("Sent joint trajectory")

def gui():
    # changed, state.teaching_mode = imgui.checkbox('Teaching mode', state.teaching_mode)
    # if changed:
    #     print("Setting teaching mode to:", state.teaching_mode)
    #     damping = np.concatenate([np.ones(5)*2, np.ones(2)*0.5])
    #     panda.teaching_mode(state.teaching_mode, dampin
    changed, state.plane_calibration = imgui.checkbox("Plane calibration", state.plane_calibration)

    if has_ros:
        if imgui.button('Test ros'):
            if itiba.Qs:
                send_joint_trajectory(itiba.Qs, itiba.dQs)
            else:
                msg_pub.publish('No joint trajectory')

    if imgui.button('Print tcp pos'):
        print(state.tcp_pos[:3])

    if imgui.button('Launch Swift'):
        itiba.launch_simulator()

    if imgui.button('Plan recorded motion'):
        Qs, dQs = editor.get_cur_motion_trajectory(robot.q)
        itiba.plan_joint_trajectory(Qs, dQs)
        state.steps, state.fk_points = compute_steps(robot, itiba.Qs)

    if imgui.button('Plan trajectory'):
        cur_pos = robot.pen_tcp_pos(robot.q)
        trajectories = pad.augment_with_pen_orientation(state.trajectories, robot, robot.qr)
        trajectories += [tu.gauss_approach(trajectories[-1][-1], cur_pos)] # [tu.approach(cur_pos, trajectories[-1][-1])]
        itiba.plan_task_trajectories(trajectories)
        state.steps, state.fk_points = compute_steps(robot, itiba.Qs)

    if imgui.button('Align tip down'):
        cur_frame = robot.pen_tcp_frame(robot.q)
        dest_frame = planner.align_frame_z(cur_frame, np.array([0, 0, -1]))

        trajectory = tu.append_rest(tu.gauss_approach(cur_frame,
                                                      dest_frame), 0.5)
        itiba.plan_task_trajectories([trajectory])
        state.steps, state.fk_points = compute_steps(robot, itiba.Qs)

    imgui.same_line()
    if imgui.button('Circle'):
        cur_frame = robot.pen_tcp_frame(robot.q)

        # Go to point above cup
        cup_frame = planner.align_frame_z(cur_frame, np.array([0, 0, -1]))
        cup_frame[:3, 3] = itiba.cup_pos
        start_dip_frame = tu.offset_frame(cup_frame, [0, 0, 120/1000])
        trajectories = [tu.trapezoidal_approach(cur_frame, start_dip_frame)]

        # Go to dipped position
        dip_frame = tu.offset_frame(cup_frame, [0, 0, 10/1000])
        trajectories += [tu.trapezoidal_approach(start_dip_frame, dip_frame, debug=True)]

        # Perform swirly motion (this approaches the side of circle)
        trajectories += [tu.xy_circles_mm(dip_frame, 10, 1.5, 4, ease_duration=1.0)]

        # Get back to point above cup
        trajectories += [tu.trapezoidal_approach(dip_frame, start_dip_frame)]

        # And back to original position
        trajectories += [tu.trapezoidal_approach(start_dip_frame, cur_frame)]

        itiba.plan_task_trajectories(trajectories)
        state.steps, state.fk_points = compute_steps(robot, itiba.Qs)

    if itiba.Qs:
        if imgui.button('Simulate trajectory'):
            itiba.simulate()

        if imgui.button("Execute trajectory"):
            imgui.open_popup('Execute')

        if imgui.begin_popup_modal('Execute').opened:
            imgui.text('Are you sure?')
            if imgui.button('OK'):
                itiba.execute()
                imgui.close_current_popup()
            imgui.same_line()
            if imgui.button('Cancel'):
                imgui.close_current_popup()
            imgui.end_popup()

    if (not state.plane_calibration):
        editor.gui(imgui)
    else:
        if imgui.button('Clear points'):
            pad.clear()

rad = geom.radians

def params_changed(*args):
    for key in args:
        if key in sketch.gui.changed:
            return True
    return False

def draw():
    background(0)
    np.random.seed(params.seed)

    # Udate drawing pad data
    pad.offset = params.drawing.plane_offset / 1000
    pad.update()

    if 'guiding' in sketch.gui.changed:
        print('Guiding params')
        panda.get_robot().set_guiding_mode([params.guiding.x,
                                            params.guiding.y,
                                            params.guiding.z,
                                            params.guiding.rot,
                                            params.guiding.rot,
                                            params.guiding.rot], params.guiding.elbow)




    params_to_check = ['seed', 'drawing.padding', 'drawing.penup_mm', 'drawing.pos_offset_x', 'drawing.pos_offset_y',
                       'traj.sigma', 'traj.isotropy', 'traj.angle']
    if params_changed(*params_to_check) or not state.trajectories:
        print('update')
        #glyphs = [tu.tsp_glyph(5, 90, 100) + [100 + i*180, 100] for i in range(3)]
        #Q = np.vstack([tu.tsp_glyph(5, 90, 100) + [100 + i*180, 100] for i in range(3)])
        pad.pos_offset = [params.drawing.pos_offset_x, params.drawing.pos_offset_y]
        pad.update()
        glyphs_3d = pad.fit_to_box(glyphs, padding=params.drawing.padding, penup_mm=params.drawing.penup_mm)
        state.trajectories = tu.smooth_strokes_lqt(glyphs_3d,
                                                   isotropy=params.traj.isotropy,
                                                   sigma=params.traj.sigma,
                                                   subsample=int(cfg.dt/cfg.robot_dt),
                                                   subd=5,
                                                   angle=params.traj.angle)

        utils.save_pkl(state.trajectories, 'trajs.pkl')
        state.trajectories = pad.fit_to_frame(state.trajectories)

        state.trajectories_2d = glyphs

    if has_robot:
        q = panda.get_robot().read_once().q # panda.q
    else:
        #q = np.array([ 4.92532656e-01, -7.48877953e-17,  3.28054421e-01, -0.00000000e+00,
        #               1.00000000e+00, -1.01393826e-17,  0.00000000e+00])
        q = robot.qr # np.zeros(7)
        #q= [-2.3774395283420826e-18, -0.2143006546605322, -8.871243762606449e-17, -2.088158222442138, 1.1696396786270597e-16, 1.8738575677821137, 0.7853981633974482]



    # P = editor.robot_polygon(q, viewport=geom.make_rect(0, 0, width, height))
    # stroke(255)
    # no_fill()
    # polyline(P, closed=False)
    robot.q = q
    state.tcp_pos = robot.pen_tcp_pos(q)
    no_fill()
    stroke(1, 0.5)
    for trajectory in state.trajectories_2d:
        draw_trajectory_2d(trajectory)
    fill(255)
    no_stroke()
    text_size(20)
    text([10, 30], '%d points'%(np.sum([len(P) for P in state.trajectories])))
    text([10, 60], 'tcp pos: [%f %f %f]'%(state.tcp_pos[0], state.tcp_pos[1], state.tcp_pos[2]))

    #v3d.sketch_view.text([2,0,0], "2m")

    #draw_robot()
    no_fill()
    editor.draw_steps(v3d, robot, [0, 0], 70, width)

def draw_gl():

    #return

    q = robot.q
    #def draw_robot(q):
    viewport = geom.make_rect(0, 0, width, height)
    v3d.set_view(viewport, [params.tsm.xoffset, params.tsm.voffset, -params.tsm.dist],
                 geom.radians(np.array([params.tsm.rotx, params.tsm.roty, params.tsm.rotz])),
                 fov=40)

    # no_fill()
    # stroke(1, 0.2)
    # stroke_weight(2.0)
    v3d.color(1, 0.2)
    v3d.line_width(2.0)
    v3d.xy_grid(2, 21)


    v3d.frame(np.eye(4), alpha=0.4, scale=0.5)


    P, penP = robot.robot_polylines(q)
    v3d.color(0, 0.5, 1.0)
    v3d.line_width(8)
    v3d.polyline(P)

    #return

    v3d.line_width(2.0)
    penframe = robot.pen_frame(q)
    v3d.frame(penframe, alpha=0.9, scale=0.05)
    v3d.frame(robot.pen_tcp_frame(q), scale=0.02)

    # hand_frame_real = robot.hand_frame(q)
    # #print('real', hand_frame_real - panda.get_pose())
    # #print('pose', panda.get_pose())
    # v3d.frame(hand_frame_real, scale=0.2)
    # hand_frame_comp = robot.pen_tcp_frame_to_hand(robot.pen_tcp_frame(q))
    # v3d.frame(hand_frame_comp, scale=0.3)

    #penframe_down = planner.align_frame_z(robot.pen_tcp_frame(q), np.array([0, 0, -1]))
    #v3d.frame(penframe_down, alpha=0.7, scale=0.05)
    #hand_frame_comp = robot.pen_tcp_frame_to_hand(penframe_down)
    #v3d.frame(hand_frame_comp, scale=0.3)

    v3d.line_width(9)
    v3d.color(1, 0.4)
    v3d.polyline(penP)
    v3d.line_width(1)
    #v3d.sketch_view.text([2,0,0], "2m")
    # P = np.random.uniform(-0.4, 0.4, (100, 3))*[1, 1, 0.01]
    # P = geom.affine_transform(geom.roty_3d(-geom.radians(params.drawing.plane_angle)), P)
    # P = P+[params.drawing.plane_dist, 0, params.drawing.plane_height]
    # v3d.color(1, 0, 1, 0.5)
    # v3d.points(P, 2)

    pad.draw(v3d)
    #return

    #state.orientation = pad.get_pen_orientation(robot, q)


    if pad.is_valid():
        #state.trajectory = pad.fit_trajectory(trajectory, padding=params.drawing.padding)

        v3d.color(1.0)
        for trajectory in state.trajectories:
            draw_trajectory_3d(v3d, trajectory)
        #v3d.polyline(state.trajectory)

        #dest = interpolate_poses(SE3(old_frame), SE3(tcp_frame), 0.5)
        #Qs = move_to_pose(robot, dest, 20)
        if itiba.Qs:
            v3d.line_width(0.3)
            v3d.color(1.0, 0.1)
            v3d.stroke_shape(state.steps)
            v3d.color(1.0, 0.0, 0.0)
            v3d.points(state.fk_points)
            #draw_steps(robot, v3d, state.Qs)

    # Reset viewport
    v3d.rend.ctx.viewport = (0, 0, sketch.window_width, sketch.window_height)
if __name__== '__main__':
    import py5canvas
    py5canvas.run()
