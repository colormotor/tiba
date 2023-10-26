#!/usr/bin/env python3
from roboticstoolbox import Robot
from polygonsoup import plut, geom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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
