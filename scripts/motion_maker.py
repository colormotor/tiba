#!/usr/bin/env python3
import numpy as np
import importlib
from motor import lqt
import imgui
from polygonsoup import utils, geom, svg, bezier, plut
import roboticstoolbox as rtb
import tiba
import pdb
import time
from importlib import reload
reload(tiba)
reload(tiba.draw)
reload(tiba.robot)
reload(tiba.tiba_sketch)
reload(bezier)

from tiba import trajectory_utils as tu
from tiba.tiba_sketch import TibaSketch
from tiba.robot import TibaRobot
from tiba.fitCurves.fitCurves import fitCurve

np.set_printoptions(suppress=True)

params = {'Seed': (5, {'min': 0, 'max': 100}),
          'svg_path': ('abr.svg', {'hide': True}),
          'GUI hue': (0.12, {'min':0, 'max': 1.0}),
          'Bezier split angle': (46.0, {'hide': True}),
          'Traj': {
              'Sigma': (0.5, {'min': 0.1, 'max':3.0}),
              'Isotropy': (0.5, {'min': 0.1, 'max':1.0}),
              'Angle': (0.0, {'min': -90, 'max':90}),
              }
          }

def random_q():
    return list(np.random.uniform(-np.pi, np.pi, 7))

#robot = TibaRobot(45, 11.5, 15) # Molotow
#robot = TibaRobot(60, 0, 0) # Molotow
# robot = TibaRobot(74, 0, 15) # Small brush
tsketch = TibaSketch()

def setup():
    # Load parameters here for standalone
    global params
    TibaSketch.add_parameters(params)
    params = sketch.parameters(params)

    #sketch.set_gui_callback(gui)
    create_canvas(1024, 768, 350) #, 300)
    set_color_scale(1.0)

    try:
        tsketch.init(sketch)
    except Exception as e:
        print(e)
        assert(False)
    load_svg(params.svg_path)

# def load_svg():
#     draft = svg.load_svg_polylines('Untitled.svg') #bezier_chains('Untitled.svg')
#     dest_rect = geom.make_rect(0, 0, width, height)
#     draft_in = geom.transform_to_rect(draft, dest_rect)
#     draft = draft_in
#     tsketch.set_draft(draft)


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
    tol = 0.1
    for chain in draft_in: #[2:4]:
        if params.bezier_split_angle > 0:
            chunks = tu.split_chain(chain, angle_tol=params.bezier_split_angle)
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
            P = tu.adaptive_subdivide(P, 5)
            draft += tu.split_polyline_at_angle(P, 160)
        #draft.append(P)
        # print('chain len:', len(chain))
        # draft.append(bezier.beziers_to_chain(beziers))

    # Plan
    try:
        if len(draft) > 1:
            draft = tu.sort_paths(draft)
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
    if imgui.button('Load SVG...'):
        path = sketch.open_file_dialog('svg', title='Open SVG file...')
        if path:
            load_svg(path)
            params.svg_path = path
    #imgui.same_line()
    changed, params.bezier_split_angle = imgui.slider_float("Angle split", params.bezier_split_angle, 0, 180)
    if changed:
        load_svg(params.svg_path)
    changed, params.config.tool = imgui.combo("Tool", params.config.tool, tsketch.get_tool_names())
    if changed:
        tsketch.set_tool(tsketch.get_tool_by_index(params.config.tool))
    tsketch.gui()

def gui_window():
    #imgui.button("Where am I")
    pass #imgui.core.show_demo_window()
#
def draw():
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
    S = geom.transform_to_rect(tsketch.draft, draft_box, 10)
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

def draw_gl():
    tsketch.draw_gl()


def mouse_pressed(button, modifiers):
    tsketch.mouse_pressed(button, modifiers)


def mouse_dragged(button, modifiers):
    tsketch.mouse_dragged(button, modifiers)


if __name__== '__main__':
    import py5canvas
    py5canvas.run()
