#%%
from build123d import *
from ocp_vscode import *

show_clear()

part = Cylinder(5, 8) - Hole(3.5, 9.0)
part.export_stl('cap.stl')
#import_stl('./panda_realsense_D4xx.stl') - Box(5,10,2)
# part = Box(5, 10, 2) - Hole(radius=2.6, depth=2)
# Cylinder()
show_all()
# %%
