#!/usr/bin/env python3
from roboticstoolbox import Robot
from polygonsoup import plut, geom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import roboticstoolbox as rtb


def esegments_2d(e, q, alpha=0.3, start=0):
    rob = Robot(e)
    print(rob.urdf_string)
    T = rob.fkine_all(q)
    segments = rob.segments()
    # draw all the line segments for the noodle plot

    for i, segment in enumerate(segments):
        linkframes = []
        for link in segment:
            if link is None:
                linkframes.append(rob.base)
            else:
                linkframes.append(T[link.number])
        points = np.array([linkframe.t for linkframe in linkframes])
        plut.stroke(points[start:], 'k', alpha=alpha)
        plut.draw_markers(points[start:], 'k', markersize=1.5, alpha=alpha)
        # self.links[i].set_xdata(points[:,0])
        # self.links[i].set_ydata(points[:,1])
    return [p.t for p in T] # T[-1]]

rad = geom.radians

def projected_robot_polygon(rob, q,
                             dist = 4.0,
                             yoffset = -0.2,
                             rotx = 275,
                             roty = 0,
                             rotz = 138,
                             viewport = [[0.0, 0.0], [1.0, 1.0]]):
    viewport = np.array(viewport)
    width, height = geom.rect_size(viewport)
    T = rob.fkine_all(q)
    P = np.array([x.t for x in T])[1:-3]
    mv = geom.trans_3d([0, yoffset, - dist]) @ (geom.rotx_3d(rad(rotx)) @
                                      geom.roty_3d(rad(roty)) @
                                      geom.rotz_3d(rad(rotz)))
    p = geom.perspective(geom.radians(20), width/height)
    S = geom.view_3d(P, mv, p, viewport=viewport)
    return S[0]

class PandaStickFigure:
    def __init__(self):
        self.robot = rtb.models.Panda()

    def polygon(self, q, viewport):
        return projected_robot_polygon(self.robot, q, viewport=viewport)
