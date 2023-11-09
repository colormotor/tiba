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

import imgui
from polygonsoup import utils, geom
import roboticstoolbox as rtb
import tiba
import time
from importlib import reload
reload(tiba)
reload(tiba.draw)
reload(tiba.robot)
from tiba.robot import Tiba
from tiba.draw import View3d, SketchView3d
from tiba import planner
import tiba.trajectory_utils as tu
reload(tu)
from spatialmath import SE3, SO3, UnitQuaternion
from spatialmath.base import trnorm
import roboticstoolbox as rtb
reload(planner)
from collections import namedtuple
import swift

params = {'Seed': (5, {'min': 0, 'max': 100}),
          'Gripper mm': (34.0, {'min': 2, 'max': 200}),
          'Gripper state': False,
          'Guiding': {
              'x': True,
              'y': True,
              'z': True,
              'rot': True,
              'elbow': True
          },
          'Drawing': {
              'Plane angle': (30, {'min': 0, 'max': 90}),
              'Plane dist' : (0.5, {'min': 0, 'max': 1}),
              'Plane height' : (0.3, {'min': 0, 'max': 1}),
              'Plane offset' : (0.0, {'min': -0.1, 'max': 0.1}),
              'Padding' : (80, {'min':0.0, 'max':450})
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


class FrankaButton:
    def __init__(self, button='circle'):
        self.button = button
        self.state = False

    def __call__(self, event):
        if not self.button in event:
            return False
        res = False
        if not self.state and event[self.button]:
            res = True
        self.state = event[self.button]
        return res

class MotionEditor:
    def __init__(self, path='motions_desk.json'):
        try:
            self.motions = utils.load_json(path)
        except:
            self.motions = {}
        self.path = path
        self.new_motion_name = ''
        self.cur_motion = -1
        self.cur_step = -1
        self.stick = tiba.draw.PandaStickFigure()

    def get_cur_motion(self):
        names = list(self.motions.keys())
        if self.cur_motion < 0 or self.cur_motion >= len(names):
            return {}
        return self.motions[names[self.cur_motion]]

    def save_motions(self):
        if not self.motions:
            return
        print(self.motions)
        utils.save_json(self.motions, self.path)

    def gui(self):
        names = list(self.motions.keys())
        if self.cur_motion >= len(names):
            self.cur_motion = -1

        clicked, self.cur_motion = imgui.listbox("Motions", self.cur_motion, names)
        if clicked:
            print(self.cur_motion)

        if imgui.button("Add"):
            imgui.open_popup('Add motion')


        if imgui.begin_popup_modal('Add motion').opened:
            imgui.text("Select name")
            changed, self.new_motion_name = imgui.input_text('Name', self.new_motion_name, 512)

            if imgui.button('OK'):
                if self.new_motion_name and self.new_motion_name not in self.motions:
                    self.motions[self.new_motion_name] = {'steps':[]}
                    self.save_motions()
                imgui.close_current_popup()
            imgui.same_line()
            if imgui.button('Cancel'):
                imgui.close_current_popup()
            imgui.end_popup()


        if self.get_cur_motion():
            imgui.same_line()
            if imgui.button("Delete"):
                self.motions.pop(names[self.cur_motion])
                self.save_motions()

            imgui.text(names[self.cur_motion])
            self.motion_gui(names[self.cur_motion])

    def add_joint_step(self, q):
        motion = self.get_cur_motion()
        if not motion:
            print('No motion selected')
            return
        motion['steps'].append({'kind':'q', 'q':list(q)})
        self.save_motions()

    def add_grip_step(self, mm):
        motion = self.get_cur_motion()
        if not motion:
            print('No motion selected')
            return
        motion['steps'].append({'kind':'grip', 'mm':mm})
        self.save_motions()

    def add_ungrip_step(self):
        motion = self.get_cur_motion()
        if not motion:
            print('No motion selected')
            return
        motion['steps'].append({'kind':'ungrip'})
        self.save_motions()

    def motion_gui(self, key):
        motion = self.motions[key]
        names = []
        for i, step in enumerate(motion['steps']):
            names.append('%d: %s'%(i+1, step['kind']))
        clicked, self.cur_step = imgui.listbox("Steps", self.cur_step, names)
        if clicked:
            print(self.cur_motion)
        if -1 < self.cur_motion < len(names):
            if imgui.button("Delete"):
                motion['steps'].pop(self.cur_step)
                self.save_motions()
        if imgui.button('Play'):
            imgui.open_popup('Play motion')
        if imgui.begin_popup_modal('Play motion').opened:
            imgui.text('Are you sure?')
            if imgui.button('OK'):
                self.play_cur_motion()
                imgui.close_current_popup()
            imgui.same_line()
            if imgui.button('Cancel'):
                imgui.close_current_popup()
            imgui.end_popup()

    def play_cur_motion(self):
        print("Playing (2 sec delay)")
        # panda.teaching_mode(False) #
        time.sleep(2.0)
        motion = self.get_cur_motion()
        if not motion:
            print('No motion')

        print('Playing')
        for step in motion['steps']:
            if step['kind'] == 'q':
                panda.move_to_joint_position(np.array(step['q']))
            elif step['kind'] == 'grip':
                mm = step['mm']

                if not state.grasped:
                    gripper.grasp(mm/1000, 0.2, 10, 0.04, 0.04)
                    state.grasped = True

            elif step['kind'] == 'ungrip':

                if state.grasped:
                    gripper.stop()
                    gripper.move(100.0/1000, 0.2)
                    state.grasped = False

        print('Ending')

    def draw_steps(self, v3d, robot, pos, size, width):
        motion = self.get_cur_motion()
        if not motion:
            return
        x, y = pos
        c = v3d.canvas()
        v3d.save_viewport()

        for i, step in enumerate(motion['steps']):
            if x + size > width:
                x = pos[0]
                y += size

            rect = geom.make_rect(x, y, size, size)
            rect = geom.scale_rect(rect, 0.95)
            if i == self.cur_step:
                c.stroke(0, 0.5, 1.0)
            else:
                c.stroke(0.5)
            c.fill(0.2, 0.5)
            c.rectangle(rect)
            v3d.set_viewport(rect)
            if step['kind'] == 'q':
                q = step['q']
                P, penP = robot.robot_polylines(q)
                v3d.color(0, 0.5, 1.0)
                v3d.line_width(3)
                v3d.polyline(P)
                v3d.polyline(penP)
            elif step['kind'] == 'grip':
                c.fill(1, 0, 0)
                c.circle(x + size/2, y+size/2, size/4)
            else:
                c.no_fill()
                c.stroke(1, 0, 0)
                c.circle(x + size/2, y+size/2, size/4)

            x += size
        v3d.restore_viewport()

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
            editor.add_joint_step(state.q)
    if check_button(event):
        if state.grasped:
            editor.add_grip_step(params.gripper_mm)
        else:
            editor.add_ungrip_step()

def grasp():
    print('grasping ', params.gripper_mm)
    gripper.grasp(params.gripper_mm/1000, 0.2, 10, 0.04, 0.04)


state = lambda: None
state.circle = False
state.grasped = False
state.q = None
state.Qs = []
state.trajectory = []
state.trajectory_2d = None
state.dt = 0.001
state.plane_calibration = False

circle_button = FrankaButton('circle')
cross_button = FrankaButton('cross')
check_button = FrankaButton('check')

if has_robot:
    hostname = '172.16.0.2'
    panda = panda_py.Panda(hostname)
    desk = panda_py.Desk(hostname, 'colormotor', 'Rapidograph22!')
    gripper = libfranka.Gripper(hostname)
    desk.listen(on_event)

editor = MotionEditor('motions_desk.json')

pad = planner.DrawingPad('pad.pkl')
pad.load()
pad.update()
pad_pose = SE3(pad.frame)
print(pad_pose)
print('t: ', pad_pose.t)
print('rpy: ', pad_pose.rpy())
print('eul: ', pad_pose.eul())


robot = Tiba(45, 11.5)
print(robot.fkine_all(robot.qr))
print(robot.q)
print(robot.qr)
robot.q = robot.qr
v3d = None

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

def interpolate_poses(p1, p2, t):
    return SE3(p1 + (p2 - p1)*t)

def linear_trajectory(P, ds):
    Q = [P[0]]
    for a, b in zip(P, P[1:]):
        nsteps = max(2, int(np.linalg.norm(a-b)/ds))
        for t in np.linspace(0, 1, nsteps)[1:]:
            Q.append(a + (b - a)*t)
    return np.array(Q)

def setup():
    sketch.set_gui_callback(gui)
    create_canvas(700, 600) #, 300)
    set_color_scale(1.0)

    global params, v3d
    params = sketch.parameters(params)
    v3d = SketchView3d(sketch)

def exit():
    print("Exit")
    if has_robot:
        desk.stop_listen()

def rest_trajectory(traj, n):
    traj = list(traj)
    for i in range(n):
        traj.append(np.array(traj[-1]))
    return traj

def gui():
    # changed, state.teaching_mode = imgui.checkbox('Teaching mode', state.teaching_mode)
    # if changed:
    #     print("Setting teaching mode to:", state.teaching_mode)
    #     damping = np.concatenate([np.ones(5)*2, np.ones(2)*0.5])
    #     panda.teaching_mode(state.teaching_mode, damping)
    changed, state.plane_calibration = imgui.checkbox("Plane calibration", state.plane_calibration)

    if imgui.button('Launch Swift'):
        env = swift.Swift()
        env.launch(realtime=True)
        env.add(robot)
    if imgui.button('Plan trajectory'):
        cur_pos = robot.pen_tcp_pos(robot.q)
        q0 = robot.q
        trajectory = [np.concatenate([pos, state.orientation]) for pos in state.trajectory]

        trajectory = rest_trajectory(planner.linear_approach(cur_pos, trajectory[0]), 400) + rest_trajectory(trajectory, 300)
        trajectory += rest_trajectory(planner.linear_approach(trajectory[-1], cur_pos), 400)
        for i in range(100):
            trajectory.append(np.array(trajectory[-1]))
        state.Qs, state.dQs = planner.plan_trajectory(robot.urdf_file,
                                                      q0,
                                                      trajectory,
                                                      dt=state.dt)
        print('Smoothing velocity')
        state.dQs = np.array(state.dQs)
        #state.dQs = geom.smoothing_spline(len(state.dQs), state.dQs, smooth_k=20)
        print("saving")
        utils.save_pkl([state.Qs, state.dQs], 'trajectory.pkl')
        # state.Qs, state.dQs = planner.plan_trajectory_constant_orientation(robot.urdf_file,
        #                                                         cur_pose,
        #                                                         state.trajectory,
        #                                                         state.orientation,
        #                                                         14,
        #                                                         state.dt,
        #                                                         q0,
        #                                                         penalty=1e-6)

    if state.Qs:
        if imgui.button('Simulate trajectory'):
            env = swift.Swift()
            env.launch(realtime=True)
            env.add(robot)

            for q in state.Qs: #dQs:
                #print(dq)
                robot.q = q
                env.step(state.dt)
        if imgui.button("Execute trajectory"):
            imgui.open_popup('Execute')

        if imgui.begin_popup_modal('Execute').opened:
            imgui.text('Are you sure?')
            if imgui.button('OK'):
                execute_trajectory()
                imgui.close_current_popup()
            imgui.same_line()
            if imgui.button('Cancel'):
                imgui.close_current_popup()
            imgui.end_popup()

    if (not state.plane_calibration):
        editor.gui()
    else:
        if imgui.button('Clear points'):
            pad.clear()

def execute_trajectory():
    print('2 second delay')
    time.sleep(2.0)
    Q = state.Qs
    dQ = state.dQs
    print('Moving to initial position')
    panda.move_to_joint_position(Q[0])
    i = 0
    print('Starting control loop')
    ctrl = controllers.JointPosition()
    panda.start_controller(ctrl)
    with panda.create_context(frequency=1000) as ctx:
        while ctx.ok() and i < len(dQ):
            ctrl.set_control(Q[i], dQ[i])
            i += 1
        print('Stopping controller')
        panda.stop_controller()
    time.sleep(1.0)
    print('Finished motion')

rad = geom.radians

def draw():

    background(0)
    np.random.seed(params.seed)

    if 'guiding' in sketch.gui.changed:
        print('Guiding params')
        panda.get_robot().set_guiding_mode([params.guiding.x,
                                            params.guiding.y,
                                            params.guiding.z,
                                            params.guiding.rot,
                                            params.guiding.rot,
                                            params.guiding.rot], params.guiding.elbow)




    if 'seed' in sketch.gui.changed or state.trajectory_2d is None:

        Q = np.vstack([tu.tsp_glyph(5, 90, 100) + [100 + i*180, 100] for i in range(3)])
        #Q = np.random.uniform(-100, 100, (6, 2)) + 100
        #Q = np.vstack([Q[0], Q, Q[-1]])
        #trajectory = geom.bspline((len(Q)-1)*900, Q)
        state.trajectory_2d = tu.smooth_stroke_lqt(Q, isotropy=0.5, sigma=1, subsample=60, angle=50)

    if has_robot:
        q = panda.get_robot().read_once().q # panda.q
    else:
        q = robot.qr # np.zeros(7)

    # P = editor.robot_polygon(q, viewport=geom.make_rect(0, 0, width, height))
    # stroke(255)
    # no_fill()
    # polyline(P, closed=False)
    robot.q = q
    state.q = q
    draw_robot(q)
    no_fill()
    editor.draw_steps(v3d, robot, [0, 0], 70, width)


def draw_robot(q):
    viewport = geom.make_rect(0, 0, width, height)
    v3d.set_view(viewport, [params.tsm.xoffset, params.tsm.voffset, -params.tsm.dist],
                 geom.radians(np.array([params.tsm.rotx, params.tsm.roty, params.tsm.rotz])))

    # no_fill()
    # stroke(1, 0.2)
    # stroke_weight(2.0)
    v3d.color(1, 0.2)
    v3d.line_width(2.0)
    v3d.xy_grid(2, 21)


    v3d.frame(np.eye(4), alpha=0.4, scale=0.5)

    P, penP = robot.robot_polylines(q)
    v3d.color(0, 0.5, 1.0)
    v3d.line_width(6)
    v3d.polyline(P)

    v3d.line_width(2.0)
    penframe = robot.pen_frame(q)
    v3d.frame(penframe, alpha=0.9, scale=0.05)
    v3d.frame(robot.pen_tcp_frame(q))

    v3d.line_width(1)
    v3d.color(1)
    v3d.polyline(penP)

    v3d.text([2,0,0], "2m")

    # P = np.random.uniform(-0.4, 0.4, (100, 3))*[1, 1, 0.01]
    # P = geom.affine_transform(geom.roty_3d(-geom.radians(params.drawing.plane_angle)), P)
    # P = P+[params.drawing.plane_dist, 0, params.drawing.plane_height]
    #v3d.color(1, 0, 1, 0.5)
    #v3d.points(P, 2)
    pad.offset = params.drawing.plane_offset
    pad.update()
    pad.draw(v3d)
    state.orientation = pad.get_pen_orientation(robot, q)

    no_fill()
    stroke(1)

    #trajectory = linear_trajectory(Q, 1)
    # trajectory = []
    # r = 100
    # for t in np.linspace(0, np.pi*2, 3000):
    #     trajectory.append([np.cos(t)*r + r, np.sin(t)*r + r])
    trajectory = state.trajectory_2d
    draw_trajectory_2d(trajectory)
    fill(255)
    no_stroke()
    text([10, 30], '%d points'%(len(trajectory)))
    if pad.is_valid():
        state.trajectory = pad.fit_trajectory(trajectory, padding=params.drawing.padding)

        v3d.color(1.0)
        draw_trajectory_3d(v3d, state.trajectory)
        #v3d.polyline(state.trajectory)

        #dest = interpolate_poses(SE3(old_frame), SE3(tcp_frame), 0.5)
        #Qs = move_to_pose(robot, dest, 20)
        if state.Qs:
            draw_steps(robot, v3d, state.Qs)

if __name__== '__main__':
    import py5canvas
    py5canvas.run()
