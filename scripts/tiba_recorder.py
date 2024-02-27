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
from tiba.tiba_sketch import imgui_text_separator
from tiba.robot import TibaRobot
from tiba import spatial, bezier, utils, svg, geom, config
reload(spatial)

from tiba.interface import FrankaButton

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

def random_q():
    return list(np.random.uniform(-np.pi, np.pi, 7))

tsketch = TibaSketch(flags=[])

def on_event(event):
    print(event)
    robot = tsketch.robot
    itiba = tsketch.itiba
    if circle_button(event):
        if itiba.grasped:
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
    panda = panda_py.Panda(hostname)
    desk = panda_py.Desk(hostname, 'colormotor', 'Rapidograph22!')
    gripper = libfranka.Gripper(hostname)
    desk.listen(on_event)
else:
    gripper = None
    desk = None
    panda = None

#robot = TibaRobot(45, 11.5, 15) # Molotow
#robot = TibaRobot(60, 0, 0) # Molotow
# robot = TibaRobot(74, 0, 15) # Small brush

state = lambda: None
state.plane_calibration = False


circle_button = FrankaButton('circle')
cross_button = FrankaButton('cross')
check_button = FrankaButton('check')


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


def exit():
    print("Exit")
    if has_robot:
        desk.stop_listen()


def gui():
    changed, state.plane_calibration = imgui.checkbox("Plane calibration", state.plane_calibration)

    if (not state.plane_calibration):
        tsketch.itiba.motion_gui(imgui)
    else:
        if imgui.button('Clear points'):
            tsketch.itiba.pad.clear()

    imgui_text_separator("Robot")
    changed, params.config.tool = imgui.combo("Tool", params.config.tool, tsketch.get_tool_names())
    if changed:
        tsketch.set_tool(tsketch.get_tool_by_index(params.config.tool))
    tsketch.gui()

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

    background(0)
    np.random.seed(params.seed)

    sketch.set_gui_theme(params.gui_hue)
    tsketch.update()

    stroke(1.0, 0.2)
    no_fill()

    tsketch.draw()

    tsketch.itiba.editor.draw_steps(tsketch.v3d, tsketch.robot, [0, height-70], 70, width)

def draw_gl():
    tsketch.draw_gl()


def mouse_pressed(button, modifiers):
    tsketch.mouse_pressed(button, modifiers)


def mouse_dragged(button, modifiers):
    tsketch.mouse_dragged(button, modifiers)


if __name__== '__main__':
    import py5canvas
    py5canvas.run()
