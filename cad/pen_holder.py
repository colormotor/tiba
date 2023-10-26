#%%
from build123d import *
from ocp_vscode import *
import numpy as np

def radians( x ):
    return np.pi/180*x

show_clear()

# For gripping
def protruberance(w, h, length):
    tri = Plane.XY*Polyline([-w*0.5, 0],
                    [w*0.5, 0],
                    [0, h], close=True)
    return Pos(0, 0, -length*0.5)*extrude(make_face(tri.edges()), amount=length)

# Slanted pen surface
def slant(w, length, angle):
    theta = radians(angle) 
    h = (length/np.cos(theta))*np.sin(theta) # np.tan(theta) 
    P = Plane.XZ*Polyline([0, 0],
                          [-length, 0],
                          [-np.cos(theta)*length, 
                           np.sin(theta)*length],
                          close=True)
    return Pos(length/2, -w/2, 0)*extrude(make_face(P.edges()), amount=w, dir=[0, 1, 0])

# Main body
w = 35
h = 50
thickness = 18
part = Box(h, w, thickness) 

# Protruberances
pw = 4
ph = 1.8
part += Pos(0, w/2, 0)*Rot(0, 90, 0)*protruberance(pw, ph, 50) # Pos(0, w/2, 0)*Box(h, 20, 5)
part += Pos(0, -w/2, 0)*Rot(0, 90, 180)*protruberance(pw, ph, 50) # Pos(0, w/2, 0)*Box(h, 20, 5)
part += Pos(0, w/2, 0)*protruberance(pw, ph, thickness) # Pos(0, w/2, 0)*Box(h, 20, 5)
part += Pos(0, -w/2, 0)*Rot(0,0,180)*protruberance(pw, ph, thickness) # Pos(0, w/2, 0)*Box(h, 20, 5)

# Slant
angle = 15  #12
if angle > 0:
    part += Pos(0, 0, thickness*0.5)*slant(w*0.7, h, angle)

# Holes for fixing AxiDraw pen holder
hole_length = thickness*5
def transform_hole(dist, voffset=0):
    return (Pos(h*0.5, 0, thickness*0.5)*
            Rot(0, angle, 0)*
            Pos(-h*0.5, 0, -hole_length*0.5)*
            Pos(dist, 0, -voffset))

part -= transform_hole(-31.8/2, 0)*Hole(3/2, hole_length) #(3/2, thickness*3)
part -= transform_hole(31.8/2, 0)*Hole(3/2, hole_length) #(3/2, thickness*3)

# Larger holes for square bolts 
screw_length = thickness*0.8

part -= transform_hole(-31.8/2, screw_length)*Box(6, 10.1, hole_length) #(3/2, thickness*3)
part -= transform_hole(31.8/2, screw_length)*Box(6, 10.1, hole_length) #(3/2, thickness*3)

def cone_slot(part, offset, base=False):
    cone_length = h*0.85
    if not base:
        part += Pos((h-cone_length)*0.5, offset, -thickness*0.0)*Rot(0, 90, 0)*Cone(1.5, 6, cone_length)
    hole_length = cone_length-2
    if base: 
        base_buffer = 0.1
    else:
        base_buffer = 0.0
    hole = Pos((h-hole_length)*0.5, offset, -thickness*0.0)*Rot(0, 90, 0)*Cone(0.5-base_buffer, 4-base_buffer, hole_length)
    if base:
        part += hole
    else:
        part -= hole
    return part

part = cone_slot(part, w*0.32)
part = cone_slot(part, -w*0.32)


part.export_stl('holder.stl')
#import_stl('./panda_realsense_D4xx.stl') - Box(5,10,2)
# part = Box(5, 10, 2) - Hole(radius=2.6, depth=2)
# Cylinder()
show_all()


#%%
base = Part()
base = cone_slot(base, w*0.32, base=True)
base = cone_slot(base, -w*0.32, base=True)
boxh = 9.5
boxl = thickness*3
base += Pos(h*0.5 + boxh*0.5, 0, -boxl/2 + thickness/2)*Box(boxh, 50, boxl)
base -= Pos(h*0.5 + boxh*0.5, 0, -boxl/2 + thickness*1.3)*Box(boxh, 14, boxl)

base -= Pos(h*0.5, w*0.45, -thickness*2) * Rot(0, 90, 0) * Hole(5/2, boxh*4)
base -= Pos(h*0.5, -w*0.45, -thickness*2) * Rot(0, 90, 0) * Hole(5/2, boxh*4)

# Slant
bz = 10
base -= Pos(h*0.5, 0, thickness/2 + bz/2)*Rot(0, angle, 0)*Box(30, 60, bz)
#base -= Pos(0, -w*0.32, 30) * Hole(5, boxh*4)

base.export_stl('base.stl')
base = Pos(0, 0, 0)*base

base.color = Color("aliceblue")
show_all()

# %%
from polygonsoup import geom, plut
import matplotlib.pyplot as plt
plut.figure((5,5))
plut.draw_line([0, 0], [-h, 0], 'r')
theta = radians(30)
plut.draw_line([0, 0], [-np.cos(theta)*h, -np.sin(theta)*h], 'b')
plut.stroke_circle([0,0], h, 'k')
plut.show()
# %%
