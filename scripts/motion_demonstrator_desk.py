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
reload(lqt)
import imgui
from polygonsoup import utils, geom, svg
import roboticstoolbox as rtb
import tiba
import time
from importlib import reload
reload(tiba)
reload(tiba.draw)
reload(tiba.robot)
from tiba.robot import Tiba
from tiba.draw import View3d, SketchView3d, GlView3d
from tiba import planner
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

np.set_printoptions(suppress=True)

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
              'Padding' : (80, {'min':0.0, 'max':450}),
              'Penup mm': (50, {'min':0.0, 'max':200})
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
        v3d = v3d.sketch_view

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
state.trajectories = []
state.trajectories_2d = []
state.dt = 0.001
state.plane_calibration = False
state.steps = []
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

editor = MotionEditor('motions_desk.json')

pad = planner.DrawingPad('pad.pkl')
pad.load()
pad.update()
pad_pose = SE3(pad.frame)
print(pad_pose)
print('t: ', pad_pose.t)
print('rpy: ', pad_pose.rpy())
print('eul: ', pad_pose.eul())


#robot = Tiba(45, 11.5) # Molotow
robot = Tiba(60, 0, 0)
print(robot.fkine_all(robot.qr))
print(robot.q)
print(robot.qr)
robot.q = robot.qr
v3d = None
glyphs = []

def compute_steps(robot, Qs, max_steps=200):
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
    v3d = GlView3d(sketch) #SketchView3d(sketch)

    global glyphs
    glyphs = svg.load_svg_polylines('tiba.svg')
    destrect = geom.make_rect(0, height-300, 400, 300)
    glyphs = geom.transform_to_rect(glyphs, destrect)

def exit():
    print("Exit")
    if has_robot:
        desk.stop_listen()

def plan_trajectory(q0, trajectory):
     state.Qs, state.dQs = planner.plan_trajectory(robot.urdf_file,
                                                      q0,
                                                      trajectory)

     utils.save_pkl([state.Qs, state.dQs], 'trajectory.pkl')
     state.steps, state.fk_points = compute_steps(robot, state.Qs)

def plan_trajectories(q0, trajectories, rest=0.5):
    ilqr = planner.iLQRPlannerVelocity(robot.urdf_file,
                                          q0)
    state.Qs, state.dQs = ilqr.plan(trajectories, rest=rest)
    utils.save_pkl([state.Qs, state.dQs], 'trajectory.pkl')
    state.steps, state.fk_points = compute_steps(robot, state.Qs)


def gui():
    # changed, state.teaching_mode = imgui.checkbox('Teaching mode', state.teaching_mode)
    # if changed:
    #     print("Setting teaching mode to:", state.teaching_mode)
    #     damping = np.concatenate([np.ones(5)*2, np.ones(2)*0.5])
    #     panda.teaching_mode(state.teaching_mode, dampinnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn
    changed, state.plane_calibration = imgui.checkbox("Plane calibration", state.plane_calibration)
    if imgui.button('Print tcp pos'):
        print(state.tcp_pos[:3])
    if imgui.button('Launch Swift'):
        env = swift.Swift()
        env.launch(realtime=True)
        env.add(robot)

    if imgui.button('Plan trajectory'):
        cur_pos = robot.pen_tcp_pos(robot.q)
        trajectories = pad.augment_with_pen_orientation(state.trajectories, robot, robot.q)

        # trajectory = tu.append_rest(trajectory, 0.5)
        # trajectory = tu.approach_inout(cur_pos, trajectory)
        # trajectory = tu.append_rest(trajectory, 0.5)

        trajectories += [tu.gauss_approach(trajectories[-1][-1], cur_pos)]
        plan_trajectories(robot.q, trajectories)

    if imgui.button('Align tip down'):
        cur_frame = robot.pen_tcp_frame(robot.q)
        dest_frame = planner.align_frame_z(cur_frame, np.array([0, 0, -1]))

        trajectory = tu.append_rest(tu.gauss_approach(cur_frame,
                                                      dest_frame), 0.5)
        plan_trajectories(robot.q, trajectory)
        print(list(state.Qs[-1]))

    imgui.same_line()
    if imgui.button('Circle'):
        cur_frame = robot.pen_tcp_frame(robot.q)
        dest_frame = planner.align_frame_z(cur_frame, np.array([0, 0, -1]))
        dest_frame[0,3] -= 0.1 # = planner.align_frame_y(dest_frame, np.array([0, 1, 0]))
        trajectory = tu.approach_inout(cur_frame,
            tu.xy_circles_mm(dest_frame, 15, 1.0, 4, ease_duration=1.0))

        plan_trajectories(robot.q, trajectory)

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

def params_changed(*args):
    for key in args:
        if key in sketch.gui.changed:
            return True
    return False

def draw():

    background(0)
    np.random.seed(params.seed)

    # Udate drawing pad data
    pad.offset = params.drawing.plane_offset
    pad.update()

    if 'guiding' in sketch.gui.changed:
        print('Guiding params')
        panda.get_robot().set_guiding_mode([params.guiding.x,
                                            params.guiding.y,
                                            params.guiding.z,
                                            params.guiding.rot,
                                            params.guiding.rot,
                                            params.guiding.rot], params.guiding.elbow)




    if params_changed('seed', 'drawing.padding', 'drawing.penup_mm') or not state.trajectories:
        print('update')
        #glyphs = [tu.tsp_glyph(5, 90, 100) + [100 + i*180, 100] for i in range(3)]
        #Q = np.vstack([tu.tsp_glyph(5, 90, 100) + [100 + i*180, 100] for i in range(3)])
        glyphs_3d = pad.fit_to_box(glyphs, padding=params.drawing.padding, penup_mm=params.drawing.penup_mm)
        state.trajectories = tu.smooth_strokes_lqt(glyphs_3d, isotropy=0.5, sigma=1, subsample=100, subd=5, angle=50)

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
    state.q = q
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

    #return

    v3d.line_width(2.0)
    penframe = robot.pen_frame(q)
    v3d.frame(penframe, alpha=0.9, scale=0.05)
    v3d.frame(robot.pen_tcp_frame(q))

    hand_frame_real = robot.hand_frame(q)
    #print('real', hand_frame_real - panda.get_pose())
    #print('pose', panda.get_pose())
    v3d.frame(hand_frame_real, scale=0.2)
    hand_frame_comp = robot.pen_tcp_frame_to_hand(robot.pen_tcp_frame(q))
    # v3d.frame(hand_frame_comp, scale=0.3)

    penframe_down = planner.align_frame_z(robot.pen_tcp_frame(q), np.array([0, 0, -1]))
    v3d.frame(penframe_down, alpha=0.7)
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

    state.orientation = pad.get_pen_orientation(robot, q)


    if pad.is_valid():
        #state.trajectory = pad.fit_trajectory(trajectory, padding=params.drawing.padding)

        v3d.color(1.0)
        for trajectory in state.trajectories:
            draw_trajectory_3d(v3d, trajectory)
        #v3d.polyline(state.trajectory)

        #dest = interpolate_poses(SE3(old_frame), SE3(tcp_frame), 0.5)
        #Qs = move_to_pose(robot, dest, 20)
        if state.Qs:
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
