#%%
print('hello')
#%%
from build123d import *
from ocp_vscode import *
import numpy as np
#%%
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

w = 35
h = 50
thickness = 18
angle = 15  #12
hole_length = thickness*5
screw_length = thickness*0.8


def make_body(has_slant=True):
    # Main body
    
    part = Box(h, w, thickness) 

    # Protruberances
    pw = 4
    ph = 1.8
    part += Pos(0, w/2, 0)*Rot(0, 90, 0)*protruberance(pw, ph, 50) # Pos(0, w/2, 0)*Box(h, 20, 5)
    part += Pos(0, -w/2, 0)*Rot(0, 90, 180)*protruberance(pw, ph, 50) # Pos(0, w/2, 0)*Box(h, 20, 5)
    part += Pos(0, w/2, 0)*protruberance(pw, ph, thickness) # Pos(0, w/2, 0)*Box(h, 20, 5)
    part += Pos(0, -w/2, 0)*Rot(0,0,180)*protruberance(pw, ph, thickness) # Pos(0, w/2, 0)*Box(h, 20, 5)

    # Slant
    
    if angle > 0 and has_slant:
        part += Pos(0, 0, thickness*0.5)*slant(w*0.7, h, angle)

    # Holes for fixing AxiDraw pen holder
    
    def transform_hole(dist, voffset=0):
        return (Pos(h*0.5, 0, thickness*0.5)*
                Rot(0, angle, 0)*
                Pos(-h*0.5, 0, -hole_length*0.5)*
                Pos(dist, 0, -voffset))

    
    # Larger holes for square bolts 
    
    if has_slant:
        part -= transform_hole(-31.8/2, 0)*Hole(3/2, hole_length) #(3/2, thickness*3)
        part -= transform_hole(31.8/2, 0)*Hole(3/2, hole_length) #(3/2, thickness*3)

        part -= transform_hole(-31.8/2, screw_length)*Box(6, 10.1, hole_length) #(3/2, thickness*3)
        part -= transform_hole(31.8/2, screw_length)*Box(6, 10.1, hole_length) #(3/2, thickness*3)

    
    part = cone_slot(part, w*0.32)
    part = cone_slot(part, -w*0.32)
    return part

part = make_body()
part.export_stl('holder.stl')
part.color = Color(1, 0.5, 1.0, 1.0)
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
base.color = Color(0, 0.8, 1.0, 1.0) #"aliceblue")
base.export_stl('base.stl')
base = Pos(0, 0, 0)*base


show_all()

#%%
measurer = Part()

padding = 2
ph = h/6
pthickness = thickness/2-padding

phh = h
measurer += Pos(phh/2 - h/2, 0, pthickness/2-thickness/2-padding)*Box(phh+padding*2, w*0.75, pthickness)
measurer -= Pos(phh/2 - h/2, 0, pthickness/2-thickness/2-padding)*Box(phh+padding*2, w*0.25, pthickness)

measurer += Pos(ph/2 - h/2, 0, pthickness/2-thickness/2-padding)*Box(ph+padding*2, w+padding*2, pthickness)
measurer += Pos(-ph/2 + h/2, 0, pthickness/2-thickness/2-padding)*Box(ph+padding*2, w+padding*2, pthickness)

flatt = thickness/2+padding
flath = h*1.7
measurer += Pos(flath/2 + h/2 - padding, 0, flatt/2 - thickness/2 - padding)*Box(flath, w*0.5, flatt)
measurer -= Pos(flath/2 + h/2 + padding*2, 0, flatt/2 - thickness/2)*Box(flath, w*0.35, flatt)
measurer -= Box(h, w, thickness) 

measurer.export_stl('measurer.stl')
measurer.color = Color(1, 0, 1, 1)
measurer = Pos(0, 0, -20)*measurer
#measurer.color = Color('red') #'aliceblue') #[1.0, 0.0, 0.0, 1.0]) #"red")


show_all()

#%%
# Probe 
probe = Part()
probe_tip_length = 10
probe_length = 50 #*0.8

probe +=  Pos(probe_length, 0, 0)*Rot(0, 90, 0)* Cylinder(4, probe_length)
probe += Pos(probe_length+probe_length*0.5+probe_tip_length*0.5, 0, 0)*Rot(0, 90, 0)*Cone(4, 1, probe_tip_length)
probe += make_body(False)
probe.color = Color(0, 0.5, 1.0, 1.0)
base.export_stl('probe.stl')
probe = Pos(0, -50, 0)*probe
show_all()
#%%
# Measureing widget
widget = Part()
widget_thick = 5
widget_w = w*0.349
widget += Box(flath*0.5, widget_w, widget_thick)
widget += Rot(0, 90, 0)*Pos(flath*0.25, 0, 5)* Box(flath*0.5, widget_w, widget_thick)
widget = Rot(90, 0, 0)*widget
widget = Pos(60, 0, widget_w*0.5-thickness*0.5)*widget
base.export_stl('widget.stl')
show_all()
#%%
batch = Part()
batch += part
batch += probe
batch += widget
batch = Pos(-150, 0, 0)*batch
batch.export_stl('batch.stl')
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


from polygonsoup import geom, plut
import matplotlib.pyplot as plt
from polygonsoup.geom import vec
plut.figure((5,5))
rect = geom.make_rect(0, -thickness/2, h, thickness)
theta = geom.radians(angle)
plut.stroke_rect(rect, 'r')
plut.draw_line([-60, 0], [60, 0], 'c', linestyle=':')
dist = 80
offset = np.tan(theta)*dist
penlength = np.cos(theta)*dist
d = vec(np.cos(theta), np.sin(-theta))
p = -geom.perp(d)
print(np.tan(theta)) 

plut.draw_line(p*offset - d*penlength, p*offset + d*penlength, 'k')
#plut.draw_line([0, 0], [-h, 0], 'r')
#theta = radians(30)
#plut.draw_line([0, 0], [-np.cos(theta)*h, -np.sin(theta)*h], 'b')
#plut.stroke_circle([0,0], h, 'k')
plut.show(axis=True)
# %%
