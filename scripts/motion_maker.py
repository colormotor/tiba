#!/usr/bin/env python3
import os
import socketserver

# OpenGL / context
os.environ["GL_CONTEXT"] = "egl"



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
from tiba.drawing_pad import DrawingPadCalibrator

reload(spatial)

np.set_printoptions(suppress=True)

has_robot = False
panda = None
gripper = None
desk = None
pad_calibrator = None
tsketch = None

# panda_loader = importlib.find_loader('panda_py')
# if panda_loader is not None:
#     import panda_py
#     from panda_py import libfranka
#     from panda_py import controllers
#     has_robot = True
# else:
#     has_robot = False

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

# def response_callback(self, data):
#     print('Received response:', data.data)

# def state_callback(data):
#     #print('Received q:', data.position)
#     if tsketch.itiba is not None:
#         tsketch.itiba.state_callback(data)

rosloader = importlib.find_loader('rospy')
if rosloader is not None:
    import rospy
    print("rospy.is_initialized():", rospy.core.is_initialized())
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String


if has_robot:
    hostname = '172.16.0.2'
    desk = panda_py.Desk(hostname, 'admin', 'adminadmin!')
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
threads = []

state = lambda: None

state.programming = True # Assume robot starts in programming mode?

def setup():
    # Load parameters here for standalone
    global params
    global tsketch


    sketch.set_gui_callback(gui)
    # create_canvas(1024, 768, 350) #, 300)
    set_color_scale(1.0)

    tsketch = TibaSketch(flags=["drawing"])
    print("Created TibaSketch")
    tsketch.init(sketch, panda=panda, gripper=gripper, init_node='tiba_motion_maker')
    print("Initialized TibaSketch")
    print("Interface object:", tsketch.itiba)
    print("itiba.has_ros():", tsketch.itiba.has_ros())
    print("itiba.msg_pub:", getattr(tsketch.itiba, "msg_pub", None))

    try:
        #tsketch.init(sketch)

        # Ensure params is a dict
        if params is None:
            params = {}

        # Add tsketch parameters to existing params dict
        params_dict = {}
        tsketch.add_parameters(params_dict)
        params.update(params_dict)
        tsketch.mark_gui_ready()

        # --- Drawing Pad Calibrator setup (if robot available) ---
        global pad_calibrator
        try:
            # Base path for drawing_pad.json (adapt if your settings path is different)
            base_path = os.path.join(tsketch.itiba.base_path, "drawing_pad") \
                if hasattr(tsketch.itiba, "base_path") \
                else os.path.join("./../robots/tiba/settings")
            # Robot object used to read Pen-TCP frame
            robot_for_calib = getattr(tsketch.itiba, "robot", None)
            pad_calibrator = DrawingPadCalibrator(base_path=base_path, robot=robot_for_calib)
            print(f"[Calib] Initialized DrawingPadCalibrator at {base_path}")
        except Exception as e:
            print("[Calib] Could not initialize DrawingPadCalibrator:", e)
            pad_calibrator = None

        # Skip tool setting for now - focus on getting GUI running
        # tool_name = tsketch.get_tool_name(params['Config']['tool'][0])
        # tsketch.set_tool(tool_name)

        # if rosloader is not None:
        #     print("Initializing ROS")
        #     try:
        #         rospy.Subscriber('/tiba/response', String, response_callback)
        #         rospy.Subscriber('/tiba/q', JointState, state_callback)
        #         tsketch.itiba.send_msg('start: Tiba sketch connecting')
        #     except Exception as e:
        #         print('Ros init failed, disabling')
        #         print(e)
        #         tsketch.itiba.has_ros = False
    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()
        #raise(e)
        #assert(False)
    # Load from params dict if path exists
    if 'path' in params:
        load(params['path'][0] if isinstance(params['path'], tuple) else params['path'])


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
    if not strokes:
        return
    chains = [np.array(P) for P in strokes['points']]
    widths = [np.array(w) for w in strokes['stroke_widths']]
    print(widths)
    if len(chains) != len(widths) or not len(widths[0].shape) or len(chains[0]) != len(widths[0]):
        widths = [np.zeros(len(P)) for P in chains]

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
    global params

    # Guard against uninitialized GUI
    if tsketch is None or tsketch.itiba is None:
        imgui.text("Initializing Interface...")
        return

    itiba = tsketch.itiba

    if imgui.button("Send test message to server"):
        print("Button clicked, itiba:", itiba)

        if not getattr(itiba, "has_ros", False):
            print("❌ ROS not connected")
        elif getattr(itiba, "msg_pub", None) is None:
            print("❌ Publisher missing")
        else:
            print("✅ Sending message...")
            itiba.send_msg("HELLO FROM MOTION_MAKER GUI")

    # Programming mode only when robot active
    if has_robot:
        changed, state.programming = imgui.checkbox("Programming mode", state.programming)
        if changed:
            tsketch.itiba.set_programming_mode(desk, state.programming)

    # --- Alte, problematische GUI (auskommentiert) ---

    # tsketch.itiba.motion_gui(imgui)
    #
    # if imgui.button('Test threads'):
    #     def thread_func1():
    #         print('Thread 1 started')
    #         n = 10
    #         config.state.progress_max = n
    #         config.state.progress = 0
    #         for i in range(n):
    #             time.sleep(0.1)
    #             config.state.progress = i + 1
    #         print('Thread 1 done')
    #
    #     def thread_func2():
    #         print('Thread 2 started')
    #         n = 15
    #         config.state.progress_max = n
    #         config.state.progress = 0
    #         for i in range(n):
    #             time.sleep(0.1)
    #             config.state.progress = i + 1
    #         print('Thread 2 done')
    #
    #     t1 = threading.Thread(target=thread_func1, name="Thread 1")
    #     t2 = threading.Thread(target=thread_func2, name="Thread 2")
    #     threads.append(t1)
    #     threads.append(t2)
    #
    # if imgui.button('Load Paths...'):
    #     path = sketch.open_file_dialog(['json', 'svg'], title='Open SVG or Json (thick) file...')
    #     if path:
    #         load(path)
    #         params.path = path
    #
    # changed, params.bezier_split_angle = imgui.slider_float(
    #     "Angle split", params.bezier_split_angle, 0, 180
    # )
    # if changed:
    #     load(params.path)
    #
    # changed, params.config.tool = imgui.combo(
    #     "Tool", params.config.tool, tsketch.get_tool_names()
    # )
    # if changed:
    #     tsketch.set_tool(tsketch.get_tool_by_index(params.config.tool))
    #
    # tsketch.gui()

    # # --- Neue, stabile GUI-Variante ---
    tsketch.itiba.motion_gui(imgui)

    if imgui.button('Load Paths...'):
        path = sketch.open_file_dialog(['json', 'svg'], title='Open SVG or Json (thick) file...')
        if path:
            load(path)
            # Falls params ein Dictionary ist:
            if isinstance(params, dict):
                params['path'] = (path, {'hide': True})
            else:
                params.path = path

    # Nutze tsketch.params (das SimpleNamespace Objekt) für die internen Werte
    tibaparams = tsketch.params if tsketch.params is not None else params

    if isinstance(tibaparams, dict):
        # Falls es ein Dict ist (von motion_maker), nutzen wir Keys
        if 'bezier_split_angle' not in tibaparams:
            tibaparams['bezier_split_angle'] = 46.0

        changed, new_angle = imgui.slider_float(
            "Angle split", tibaparams['bezier_split_angle'], 0, 180
        )
        if changed:
            tibaparams['bezier_split_angle'] = new_angle
            if 'path' in tibaparams:
                load(tibaparams['path'][0] if isinstance(tibaparams['path'], tuple) else tibaparams['path'])
    else:
        # Falls es ein SimpleNamespace ist (von tiba_sketch)
        if not hasattr(tibaparams, "bezier_split_angle"):
            tibaparams.bezier_split_angle = 46.0
        changed, tibaparams.bezier_split_angle = imgui.slider_float(
            "Angle split", tibaparams.bezier_split_angle, 0, 180
        )
        if changed and hasattr(params, 'path'):
            load(params.path)

    # Tool-Auswahl
    tool_names = tsketch.get_tool_names()

    # Robuster Zugriff auf das aktuelle Tool
    current_tool_idx = 0
    if isinstance(tibaparams, dict) and 'Config' in tibaparams:
        current_tool_idx = tibaparams['Config']['tool'][0]
    elif hasattr(tibaparams, 'config'):
        current_tool_idx = tibaparams.config.tool

    changed, new_tool = imgui.combo("Tool", current_tool_idx, tool_names)
    if changed:
        if isinstance(tibaparams, dict) and 'Config' in tibaparams:
            tibaparams['Config']['tool'] = (new_tool, tibaparams['Config']['tool'][1])
        elif hasattr(tibaparams, 'config'):
            tibaparams.config.tool = new_tool
        tsketch.set_tool(tsketch.get_tool_by_index(new_tool))

    if hasattr(tsketch.itiba, "pad_calibration_gui"):
        tsketch.itiba.pad_calibration_gui(imgui)

    if tsketch.itiba is not None and hasattr(tsketch.itiba, "tool_rack_gui"):
        tsketch.itiba.tool_rack_gui(imgui)

    if tsketch.itiba is not None and hasattr(tsketch.itiba, "color_palette_gui"):
        tsketch.itiba.color_palette_gui(imgui)

    # WICHTIG: Die Haupt-GUI von TibaSketch aufrufen
    # Diese Funktion setzt intern _gui_ready auf True, wenn sie durchläuft
    tsketch.gui()

def gui_window():
    #imgui.button("Where am I")
    # imgui.core.show_demo_window()
    pass

def draw():
    if has_robot:
        tsketch.itiba.read_states()

    if len(threads):
        current = threads[0]
        if current.ident is None:
            imgui.open_popup("Progress")
            config.state.progress_text = current.name
            current.start()
        elif not current.is_alive():
            current.join()
            threads.pop(0)
            print('Finished thread, %d left', len(threads))


    imgui.set_next_window_size(200, 70)
    if imgui.begin_popup_modal("Progress").opened:
        imgui.text(config.state.progress_text)
        imgui.progress_bar(config.state.progress / config.state.progress_max)

        if config.state.progress >= config.state.progress_max:
            imgui.close_current_popup()
        imgui.end_popup()

    background(0)

    # robustes Seed-Handling
    seed = 5
    if isinstance(params, dict) and 'Seed' in params:
        val = params['Seed']
        seed = val[0] if isinstance(val, tuple) else val
    np.random.seed(seed)

    # robustes GUI-Theme-Handling
    gui_hue = 0.12
    if isinstance(params, dict) and 'GUI hue' in params:
        val = params['GUI  hue']
        gui_hue = val[0] if isinstance(val, tuple) else val
    sketch.set_gui_theme(gui_hue)

    # Robuste Prüfung vor tsketch.update()
    if tsketch.params is None:
        return

    if not hasattr(tsketch.params, 'config') or tsketch.params.config is None:
        return

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
        text([10, 70 + i*20], ['x', 'y', 'z'][i] + ': [%.2f %.2f %.2f]'%tuple(fr[:3, i]))


def draw_gl():
    # tsketch.draw_gl(
    pass


def mouse_pressed(button, modifiers):
    tsketch.mouse_pressed(button, modifiers)


def mouse_dragged(button, modifiers):
    tsketch.mouse_dragged(button, modifiers)


if __name__== '__main__':
    import py5canvas
    py5canvas.run()
