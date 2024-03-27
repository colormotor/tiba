#!/usr/bin/env python3
import numpy as np
from sensor_msgs.msg import Image as msg_Image
import rospy
import imgui
params = {'Test':(1.0, {'min':0.0, 'max':1.0})}
params = sketch.parameters(params)

state = lambda: None
state.image = None

def got_image(data):
    im = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
    state.image = im
    #print(data.shape)

rospy.init_node('camera_viewer', anonymous=True)
rospy.Subscriber('/camera/color/image_raw', msg_Image, got_image)

def gui():
    if imgui.button('Save video'):
        sketch.grab_movie('draw.mp4', 1500)
    pass


def setup():
    create_canvas(640, 480)
    set_color_scale(1)
    frame_rate(30)

def draw():
    background(0)
    if state.image is not None:
        image(state.image) #, width, height*state.image.shape[0]/state.image.shape[1])
