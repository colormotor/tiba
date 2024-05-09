#!/usr/bin/env python3
from importlib import reload

import threading

import numpy as np
import importlib
from tiba import lqt
import imgui
# from tiba import utils, geom, svg
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
from tiba.fitCurves.fitCurves import fitCurve
from tiba import spatial, bezier, utils, svg, geom, config
reload(spatial)

np.set_printoptions(suppress=True)

panda_loader = importlib.find_loader('panda_py')
if panda_loader is not None:
    import panda_py
    from panda_py import libfranka
    from panda_py import controllers
    has_robot = True
else:
    has_robot = False

params = {'Seed': (5, {'min': 0, 'max': 100}),
          'path': ('abr.svg', {'hide': True}),
          'GUI hue': (0.12, {'min':0, 'max': 1.0}),
          'Bezier split angle': (46.0, {'hide': True}),
          'Traj': {
              'Sigma': (0.5, {'min': 0.1, 'max':13.0}),
              'Isotropy': (0.5, {'min': 0.1, 'max':1.0}),
              'Angle': (0.0, {'min': -90, 'max':90}),
              }
          }

def random_q():
    return list(np.random.uniform(-np.pi, np.pi, 7))

#robot = TibaRobot(45, 11.5, 15) # Molotow
#robot = TibaRobot(60, 0, 0) # Molotow
# robot = TibaRobot(74, 0, 15) # Small brush

def response_callback(self, data):
    print('Received response:', data.data)

def state_callback(data):
    #print('Received q:', data.position)
    if tsketch.itiba is not None:
        tsketch.itiba.state_callback(data)

rosloader = importlib.find_loader('rospy')
if rosloader is not None:
    import rospy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String


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

flags = ['drawing']
if has_robot:
    flags.append('no_server')
tsketch = TibaSketch(flags=flags)
threads = []

state = lambda: None

state.programming = True # Assume robot starts in programming mode?

def setup():
    # Load parameters here for standalone
    global params
    tsketch.add_parameters(params)
    params = sketch.parameters(params)

    #sketch.set_gui_callback(gui)
    create_canvas(1024, 768, 350) #, 300)
    set_color_scale(1.0)

    try:
        
        tsketch.init(sketch)
        tsketch.init(sketch, panda=panda, gripper=gripper, init_node='tiba_motion_maker')
        
        if rosloader is not None:
            print("Initializing ROS")
            try:
                rospy.Subscriber('/tiba/response', String, response_callback)
                rospy.Subscriber('/tiba/q', JointState, state_callback)
                tsketch.itiba.send_msg('start: Tiba sketch connecting')
            except Exception as e:
                print(e)
                tsketch.itiba.has_ros = False

    except Exception as e:
        print(e)
        # assert(False)
    load(params.path) #params.svg_path)

# def load_svg():
#     draft = svg.load_svg_polylines('Untitled.svg') #bezier_chains('Untitled.svg')
#     dest_rect = geom.make_rect(0, 0, width, height)
#     draft_in = geom.transform_to_rect(draft, dest_rect)
#     draft = draft_in
#     tsketch.set_draft(draft)

def load(path):
    if '.svg' in path:
        load_svg(path)
    else:
        load_thick_paths(path)
        
def load_thick_paths(path):
    strokes = utils.load_json(path)
    chains = [np.array(P) for P in strokes['points']]
    widths = [np.array(w) for w in strokes['stroke_widths']]
    tsketch.set_draft([np.hstack([P, w.reshape(-1,1)]) for P, w in zip(chains, widths)])

def load_svg(path):
    print('Loading SVG:', path)
    try:
        draft = svg.load_svg_bezier_chains(path)
    except FileNotFoundError as e:
        print(e)
        return
    zz = svg.load_svg_polylines(path)

    dest_rect = geom.make_rect(0, 0, width, height)
    draft_in = geom.transform_to_rect(draft, dest_rect)
    draft = []
    tol = 0.2
    for chain in draft_in: #[2:4]:
        if params.bezier_split_angle > 0:
            chunks = bezier.split_chain(chain, angle_tol=params.bezier_split_angle)
        else:
            chunks = [chain]
        chunk_beziers = []
        for chunk in chunks:
            chunk_beziers.append(chunk)
            # pts = bezier.cubic_bezier_adaptive(chunk, tol)
            #pts = bezier.bezier_piecewise(chunk, 30)
            # beziers = fitCurve(pts, 4)
            # chunk_beziers += beziers #.append(beziers)

        for chain in chunk_beziers:
            #chain = bezier.beziers_to_chain(chunk_beziers)
            P = bezier.cubic_bezier_adaptive(chain, tol)
            #P = geom.adaptive_subdivide(P, 5)
            draft += [P] # geom.split_polyline_at_angle(P, 160)
        #draft.append(P)
        # print('chain len:', len(chain))
        # draft.append(bezier.beziers_to_chain(beziers))

    # Plan
    try:
        if len(draft) > 1:
            # Sort from top left first
            #draft = sorted(draft, key=lambda P: np.min(P[:, 1]) + np.min(P[:, 0]))
            #draft = spatial.sort_paths(draft, reversable=True, join_thresh=4)
            pass
        tsketch.set_draft(draft)
    except (RuntimeWarning, ValueError) as e:
        print('Sort paths failed')
        print(e)
        print([np.any(np.isnan(P)) for P in draft])
        print([np.min(geom.chord_lengths(P)) for P in draft])
    print(geom.bounding_box(draft))
    tsketch.set_draft(draft)


def exit():
    print("Exit")


def gui():
    if has_robot:
        changed, state.programming = imgui.checkbox("Programming mode", state.programming)
        if changed:
            tsketch.itiba.set_programming_mode(desk, state.programming)

    tsketch.itiba.motion_gui(imgui)
    
    if imgui.button('Test threads'):
        def thread_func1():
            print('Thread 1 started')
            n = 10
            config.state.progress_max = n
            config.state.progress = 0
            for i in range(n):
                time.sleep(0.1)
                config.state.progress = i+1

            print('Thread 1 done')
        def thread_func2():
            print('Thread 2 started')
            n = 15
            config.state.progress_max = n
            config.state.progress = 0
            for i in range(n):
                time.sleep(0.1)
                config.state.progress = i+1

            print('Thread 2 done')
        t1 = threading.Thread(target=thread_func1, name="Thread 1")
        t2 = threading.Thread(target=thread_func2, name="Thread 2")
        threads.append(t1)
        threads.append(t2)
    if imgui.button('Load SVG...'):
        path = sketch.open_file_dialog('svg', title='Open SVG file...')
        if path:
            load_svg(path)
            params.path = path
    if imgui.button('Load thick...'):
        path = sketch.open_file_dialog('json', title='Open thick path file...')
        if path:
            load(path)
            params.path = path
    #imgui.same_line()
    changed, params.bezier_split_angle = imgui.slider_float("Angle split", params.bezier_split_angle, 0, 180)
    if changed:
        load(params.path)
    changed, params.config.tool = imgui.combo("Tool", params.config.tool, tsketch.get_tool_names())
    if changed:
        tsketch.set_tool(tsketch.get_tool_by_index(params.config.tool))
    tsketch.gui()

def gui_window():
    #imgui.button("Where am I")
    # imgui.core.show_demo_window()
    pass

def draw():
    if has_robot:
        tsketch.itiba.read_states()

    if len(threads):
        if threads[0].ident is None:
            imgui.open_popup("Progress")
            config.state.progress_text = threads[0].name
            threads[0].start()
        elif not threads[0].is_alive():
            threads[0].join()
            threads.pop(0)
            print('Finished thread, %d left', len(threads))
            if threads:
                print("Starting new thread")
                config.state.progress_text = threads[0].name
                threads[0].start()

    imgui.set_next_window_size(200, 70)
    if imgui.begin_popup_modal("Progress").opened:
        imgui.text(config.state.progress_text)
        imgui.progress_bar(config.state.progress / config.state.progress_max)

        if config.state.progress >= config.state.progress_max:
            imgui.close_current_popup()
        imgui.end_popup()

    background(0)
    np.random.seed(params.seed)

    sketch.set_gui_theme(params.gui_hue)
    tsketch.update()

    stroke(1.0, 0.2)
    no_fill()

    draft_box = geom.make_rect(width-256, 0, 256, 256)
    fill(1, 0.5)
    rectangle(draft_box)
    no_fill()

    draft, trajectories = tsketch.get_fitted_draft_and_trajectories()

    draft_2d = [P[:,:2] for P in draft]
    mat = geom.rect_in_rect_transform(geom.bounding_box(draft_2d), draft_box, 10)
    S = geom.affine_transform(mat, draft_2d)
    for i, P in enumerate(S):
        stroke(0, 0.5)
        polyline(P)

    
    if trajectories is not None:
        trajectories_2d = [P[:,:2] for P in trajectories]
        S = geom.affine_transform(mat, trajectories_2d)
        for i, P in enumerate(S):
            stroke(0) #*plut.default_color(i), 0.5)
            polyline(P)

    stroke(1.0, 0.0, 0.0, 0.2)
    if len(S) > 0:
        for P, Q in zip(S, S[1:]):
            line(P[-1], Q[0])
    fill(1)
    no_stroke()
    text_size(16)
    for i, P in enumerate(S):
        text(P[0], str(i))

        #polyline(bezier.bezier_piecewise(P)) # + np.random.uniform(-30, 30, 2), 3))

    tsketch.draw()

    fr = tsketch.robot.pen_tcp_frame()
    pose = tsketch.robot.pen_tcp_pose()
    #print(np.array(pose))
    #print('Frame is valid:', SE3.isvalid(np.array(pose)))
    text_size(15)
    fill(1)
    for i in range(3):
        text([10, 60 + i*20], ['x', 'y', 'z'][i] + ': [%.2f %.2f %.2f]'%tuple(fr[:3, i]))


def draw_gl():
    tsketch.draw_gl()


def mouse_pressed(button, modifiers):
    tsketch.mouse_pressed(button, modifiers)


def mouse_dragged(button, modifiers):
    tsketch.mouse_dragged(button, modifiers)


if __name__== '__main__':
    import py5canvas
    py5canvas.run()
