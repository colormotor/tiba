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
from tiba import trajectory_utils as tu
from tiba.tiba_sketch import TibaSketch
from tiba.robot import TibaRobot
from tiba.fitCurves.fitCurves import fitCurve

np.set_printoptions(suppress=True)

params = {'Seed': (5, {'min': 0, 'max': 100}),
          'Traj': {
              'Sigma': (0.5, {'min': 0.1, 'max':3.0}),
              'Isotropy': (0.5, {'min': 0.1, 'max':1.0}),
              'Angle': (0.0, {'min': -90, 'max':90}),
              }
          }

TibaSketch.add_parameters(params)
params = sketch.parameters(params)


def random_q():
    return list(np.random.uniform(-np.pi, np.pi, 7))

robot = TibaRobot(45, 11.5, 15) # Molotow
#robot = TibaRobot(60, 0, 0) # Molotow
# robot = TibaRobot(74, 0, 15) # Small brush
tsketch = TibaSketch()

def setup():
    sketch.set_gui_callback(gui)
    create_canvas(700, 600) #, 300)
    set_color_scale(1.0)

    tsketch.init(sketch, robot, 'bezier')
    load_svg()

# def load_svg():
#     draft = svg.load_svg_polylines('Untitled.svg') #bezier_chains('Untitled.svg')
#     dest_rect = geom.make_rect(0, 0, width, height)
#     draft_in = geom.transform_to_rect(draft, dest_rect)
#     draft = draft_in
#     tsketch.set_draft(draft)

def load_svg():
    draft = svg.load_svg_bezier_chains('abr.svg')
    dest_rect = geom.make_rect(0, 0, width, height)
    draft_in = geom.transform_to_rect(draft, dest_rect)
    draft = []
    for chain in draft_in[:]:
        chunks = tu.split_chain(chain, 10)
        chunk_beziers = []
        for chunk in chunks:
            pts = bezier.bezier_piecewise(chunk, 10)
            beziers = fitCurve(pts, 10)
            chunk_beziers += beziers #.append(beziers)
        draft.append(bezier.beziers_to_chain(chunk_beziers))
            #draft.append(bezier.beziers_to_chain(beziers))

    tsketch.set_draft(draft)

def exit():
    print("Exit")


def gui():
    tsketch.gui()


def draw():
    background(0)
    np.random.seed(params.seed)

    tsketch.update()

    stroke(1.0, 0.2)
    no_fill()
    for i, P in enumerate(tsketch.draft):
        stroke(*plut.default_color(i), 0.2)
        polyline(P)
        #polyline(bezier.bezier_piecewise(P)) # + np.random.uniform(-30, 30, 2), 3))


def draw_gl():
    tsketch.draw_gl()


if __name__== '__main__':
    import py5canvas
    py5canvas.run()
