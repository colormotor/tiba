#!/usr/bin/env python3
# HAND COORDS!!!
drop_pos = np.array([0.21464598888936506, 0.45755225727463517, 0.12085804386599398])
up = np.array([0, 0, 1])

start_offset = 150/1000
drop_offset = 10/1000
joint_move('Tool drop start') #side drop')
hand_align_pick_tool()
goto_xyz_hand(drop_pos)
goto_xyz_hand(drop_pos - up*drop_offset)
ungrasp()
goto_xyz_hand(drop_pos + up*start_offset)
joint_move('Home')
