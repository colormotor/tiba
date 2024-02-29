#!/usr/bin/env python3
from importlib import reload

import threading

import numpy as np
import importlib
from motor import lqt
import imgui
# from polygonsoup import utils, geom, svg
import roboticstoolbox as rtb
import tiba
import pdb
import time

reload(tiba)
reload(tiba.draw)
reload(tiba.robot)
reload(tiba.tiba_sketch)

from tiba import trajectory_utils as tu
from tiba.tiba_sketch import TibaSketch
from tiba.robot import TibaRobot
from tiba import spatial, bezier, utils, svg, geom, config, ui
reload(spatial)
from tiba.interface import FrankaButton

from spatialmath import SE3, SO3, UnitQuaternion
from spatialmath.base import trnorm

reload(tu)
reload(ui)

panda_loader = importlib.find_loader('panda_py')
if panda_loader is not None:
    import panda_py
    from panda_py import libfranka
    from panda_py import controllers
    has_robot = True
else:
    has_robot = False

np.set_printoptions(suppress=True)

params = {'Seed': (5, {'min': 0, 'max': 100}),
          'GUI hue': (0.12, {'min':0, 'max': 1.0}),
          'Gripper mm': (34.0, {'min': 2, 'max': 200}),
          'Guiding': {
              'x': (True, {'sameline':0}),
              'y': (True, {'sameline':1}),
              'z': (True, {'sameline':1}),
              'rot': (True, {'sameline':0}),
              'elbow': (True, {'sameline':1})
          },
          }

button_color = [1.0, 0.5, 0.0, 1.0]

def random_q():
    return list(np.random.uniform(-np.pi, np.pi, 7))

tsketch = TibaSketch(flags=['no_server'])

def on_event(event):
    print(event)
    robot = tsketch.robot
    itiba = tsketch.itiba
    print('Grasped: ', itiba.grasped)
    if circle_button(event):
        if not itiba.grasped:
            print('Grasp')
            itiba.grasp(params.gripper_mm)
        else:
            print('Ungrasp')
            itiba.ungrasp()
    if cross_button(event):
        if state.plane_calibration:
            p = robot.pen_tcp_pos(robot.q)[:3]
            tsketch.pad.add_point(p)
        else:
            tsketch.editor.add_joint_step(robot.q)
    if check_button(event):
        if itiba.grasped:
            tsketch.editor.add_grip_step(params.gripper_mm)
        else:
            tsketch.editor.add_ungrip_step()

if has_robot:
    hostname = '172.16.0.2'
    desk = panda_py.Desk(hostname, 'colormotor', 'Rapidograph22!')
    print("Created desk")
    if not desk.has_control():
        print("Forcing control")
        desk.take_control(True)
        desk.activate_fci()
    panda = panda_py.Panda(hostname)
    gripper = libfranka.Gripper(hostname)

else:
    gripper = None
    desk = None
    panda = None

#robot = TibaRobot(45, 11.5, 15) # Molotow
#robot = TibaRobot(60, 0, 0) # Molotow
# robot = TibaRobot(74, 0, 15) # Small brush

state = lambda: None
state.plane_calibration = False
state.programming = True # Assume robot starts in programming mode?
state.dest_pose = None
state.move_amount = np.zeros(3)

circle_button = FrankaButton('circle')
cross_button = FrankaButton('cross')
check_button = FrankaButton('check')

def check_collision(q, itiba):
    r = itiba.collision_robot
    r.q = q
    collision_links = [link for link in r.links if link.collision][3:] #[4:-1]
    print([l.name for l in collision_links])
    n = len(collision_links)
    for i in range(n):
        stop = False
        for j in range(i+2, n):
            isCollide = collision_links[i].iscollided(shape = collision_links[j].collision[0])
            if isCollide:
                stop = True
                print("--COLLISION DETECT:",collision_links[i].name," will collide with ",collision_links[j].name)
                break
        if stop: break
    return stop


def setup():
    # Load parameters here for standalone
    global params
    tsketch.add_parameters(params)
    params = sketch.parameters(params)

    #sketch.set_gui_callback(gui)
    create_canvas(1024, 768, 350) #, 300)
    set_color_scale(1.0)

    try:
        tsketch.init(sketch, panda=panda, gripper=gripper)
    except Exception as e:
        print(e)
        assert(False)

    # # Dip pos
    # tsketch.robot.q = [                   -1.2621834561240015,
    #                 -0.5742350572707479,
    #                 -0.04934481753466121,
    #                 -2.510081760803854,
    #                 -0.014960459649711948,
    #                 2.1480748627228876,
    #                 0.9754806254955254]

    # # Pick pens
    # tsketch.robot.q = [
    #                   0.8923272938377215,
    #                 -0.3647182149714724,
    #                 0.26038909909143254,
    #                 -2.45132644355543,
    #                 0.07505659465603819,
    #                 2.155968632893118,
    #                 0.8315813678712944
    #             ]
    #
    # tsketch.robot.q = [
    #                      0.8899934283549032,
    #                 0.26157877084836123,
    #                 0.22606831535379185,
    #                 -2.5866245919384454,
    #                 -0.20197267653590556,
    #                 2.855830673453475,
    #                 -2.6059172405630866
    # ]
    # Self collision

    #tsketch.robot.q = [-1.639689114087109, -1.7716644918674729, 0.007238895131838701, -3.013367922364223, -0.015000028089361048, 0.9541045565059841, 0.7067115097185542]

    check_collision(tsketch.robot.q, tsketch.itiba)

    if has_robot:
        print("Start listen")
        desk.listen(on_event)
    print("End init")


def exit():
    print("Exit")
    tsketch.release()
    if has_robot:
        #desk.release_control()
        print("Stopping listen")
        if desk._listening:
            desk.stop_listen()
    print("End exit")

def gui():
    if has_robot:
        changed, state.programming = imgui.checkbox("Programming mode", state.programming)
        if changed:
            tsketch.itiba.set_programming_mode(desk, state.programming)

    changed, state.plane_calibration = imgui.checkbox("Plane calibration", state.plane_calibration)

    if (not state.plane_calibration):
        tsketch.itiba.motion_gui(imgui)
    else:
        if imgui.button('Clear points'):
            tsketch.itiba.pad.clear()

    ui.imgui_text_separator("Robot")
    changed, params.config.tool = imgui.combo("Tool", params.config.tool, tsketch.get_tool_names())
    if changed:
        tsketch.set_tool(tsketch.get_tool_by_index(params.config.tool))

    ui.imgui_text_separator("Scripts")
    script = tsketch.script_gui()
    if script:
        args = script['args'] if 'args' in script else {}
        tsketch.plan_script(script['file'], **args)
        if tsketch.itiba.Qs:
            print('Script plan sizes:', len(tsketch.itiba.Qs), len(tsketch.itiba.dQs))
            tsketch.itiba.plot_trajectory_stats()
    ui.imgui_text_separator("Motion")
    if imgui.button('Align hand for tool'):
        #tsketch.itiba.set_programming_mode(desk, False)
        robot = tsketch.robot
        cur_pos = robot.get_pos(None, 'tiba_hand_tcp')
        # cur_frame = robot.pen_tcp_frame()
        # dest_frame = np.array(cur_frame)
        # dest_frame[:3, 0] = [0, -1, 0]
        # dest_frame[:3, 1] = [-1, 0, 0]
        # dest_frame[:3, 2] = [0, 0, -1]
        # dest_frame = tu.trnorm(dest_frame)
        # print(dest_frame)

        #dest_pos = tu.to_pose(cur_frame)
        #dest_pos = align_pos_zy(cur_pos, [0, 0, -1], [1, 0, 0]) # On right (dip)
        #tu.to_pose(cur_pos) #cur_frame #tu.to_pose(
        dest_pos = tu.align_pos_zy(cur_pos, [0, 0, -1], [-1, 0, 0]) # On left (pick pens)
        cur_frame = robot.pen_tcp_pose() #robot.hand_frame_to_pen_tcp(tu.to_pose(cur_pos))
        dest_frame = robot.hand_frame_to_pen_tcp(tu.to_pose(dest_pos))

        trajectories = [tu.approach_sequence([cur_frame, dest_frame])]
        tsketch.plan_task_trajectories(trajectories)
        tsketch.itiba.plot_trajectory_stats() #self.task_trajectories_augmented)
        #
    if imgui.button('Align pen for dip'):
        #tsketch.itiba.set_programming_mode(desk, False)
        robot = tsketch.robot
        cur_pos = robot.pen_tcp_pos()
        # cur_frame = robot.pen_tcp_frame()
        # dest_frame = np.array(cur_frame)
        # dest_frame[:3, 0] = [0, -1, 0]
        # dest_frame[:3, 1] = [-1, 0, 0]
        # dest_frame[:3, 2] = [0, 0, -1]
        # dest_frame = tu.trnorm(dest_frame)
        # print(dest_frame)
        #dest_pos = tu.to_pose(cur_frame)
        #dest_pos = align_pos_zy(cur_pos, [0, 0, -1], [1, 0, 0]) # On right (dip)
        dest_pos = tu.align_pos_zy(cur_pos, [0, 0, -1], [-1, 0, 0]) # On left (pick pens)

        trajectories = [tu.approach_sequence([cur_pos, dest_pos])]
        tsketch.plan_task_trajectories(trajectories)
        tsketch.itiba.plot_trajectory_stats() #self.task_trajectories_augmented)
        #tsketch.itiba.set_programming_mode(desk, state.programming)

    # if imgui.button('Try script'):
    #     #tsketch.plan_script('motions/dip.py', dip_radius_mm=7.5, clean_radius_mm=34.0) #dip_radius_mm=7.5, clean_radius_mm=14.0)
    #     tsketch.plan_script('motions/pick.py')
    #     if tsketch.itiba.Qs:
    #         tsketch.itiba.plot_trajectory_stats()

    dest_position = None

    for i in range(3):
        _, state.move_amount[i] = imgui.slider_float("Move %s mm"%['X', 'Y', 'Z'][i], state.move_amount[i], -300, 300.0)

    dest_frame = np.array(tsketch.robot.pen_tcp_frame())
    dest_frame[:3, 3] += state.move_amount/1000.0
    #print(dest_frame)
    state.dest_pose = tu.to_pose(dest_frame)

    if np.linalg.norm(state.move_amount) > 0:
        dest_frame = np.array(tsketch.robot.pen_tcp_frame())
        if imgui.button('Make Move'):
            dest_position = state.dest_pose
            state.move_amount = np.zeros(3)
    if tsketch.itiba.editor.get_cur_position():
        if ui.imgui_color_button('Cartesian move to position'):
            q = tsketch.itiba.editor.get_cur_position()
            dest_position = tsketch.robot.pen_tcp_pos(q)

        if ui.imgui_color_button('Joint move to position'):
            q = tsketch.itiba.editor.get_cur_position()
            tsketch.plan_joint_approach(q)
            tsketch.itiba.plot_trajectory_stats() #self.task_trajectories_augmented)

    if dest_position is not None:
        cur_pos = tsketch.robot.pen_tcp_pos()
        trajectories = [tu.approach_sequence([cur_pos, dest_position])]
        tsketch.plan_task_trajectories(trajectories)
        tsketch.itiba.plot_trajectory_stats() #self.task_trajectories_augmented)
    tsketch.gui()

    if tsketch.itiba.Qs:
        imgui.separator()
        if not has_robot:
            if ui.imgui_color_button('Simulate trajectory'):
                #self.threads.append(threading.Thread(target=self.itiba.simulate, name='nogui'))
                tsketch.itiba.simulate()
                tsketch.robot.q = tsketch.itiba.Qs[-1]
        else:
            if imgui.button('Simulate trajectory'):
                tsketch.itiba.simulate()
                if not has_robot:
                    tsketch.robot.q = tsketch.itiba.Qs[-1]

            if ui.imgui_color_button("Execute trajectory"):
                imgui.open_popup('Execute')

            if imgui.begin_popup_modal('Execute').opened:
                imgui.text('Are you sure?')
                if imgui.button('OK'):
                    tsketch.itiba.set_programming_mode(desk, False)
                    tsketch.itiba.execute()
                    # tsketch.itiba.set_programming_mode(desk, state.programming) #
                    imgui.close_current_popup()
                imgui.same_line()
                if imgui.button('Cancel'):
                    imgui.close_current_popup()
                imgui.end_popup()

    if 'guiding' in sketch.gui.changed:
        print('Guiding params')
        panda.get_robot().set_guiding_mode([params.guiding.x,
                                            params.guiding.y,
                                            params.guiding.z,
                                            params.guiding.rot,
                                            params.guiding.rot,
                                            params.guiding.rot], params.guiding.elbow)

def gui_window():
    #imgui.button("Where am I")
    # imgui.core.show_demo_window()
    pass

def draw():
    if has_robot:
        tsketch.itiba.read_states()



    #print(desk._token.id, desk._get_active_token()) #has_control())
    background(0)
    np.random.seed(params.seed)

    sketch.set_gui_theme(params.gui_hue)
    tsketch.update()

    stroke(1.0, 0.2)
    no_fill()

    tsketch.draw()

    fr = tsketch.robot.pen_tcp_frame()
    pose = tsketch.robot.pen_tcp_pose()
    #print(np.array(pose))
    #print('Frame is valid:', SE3.isvalid(np.array(pose)))
    text_size(15)
    fill(1)
    for i in range(3):
        text([10, 60 + i*20], ['x', 'y', 'z'][i] + ': [%.2f %.2f %.2f]'%tuple(fr[:3, i]))


    tsketch.itiba.editor.draw_steps(tsketch.v3d, tsketch.robot, [0, height-70], 70, width)


def draw_gl():
    v3d = tsketch.v3d
    tsketch.begin_draw_gl()
    tsketch.draw_gl_main()

    if state.dest_pose is not None:
        v3d.frame(np.array(state.dest_pose), alpha=0.9, scale=0.1)

    tsketch.end_draw_gl()
def mouse_pressed(button, modifiers):
    tsketch.mouse_pressed(button, modifiers)


def mouse_dragged(button, modifiers):
    tsketch.mouse_dragged(button, modifiers)


if __name__== '__main__':
    import py5canvas
    py5canvas.run()
