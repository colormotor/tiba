#!/usr/bin/env python3
# Initial joint configuration
start_q = np.array([0.9944783904786348, -0.027600061799658623, 0.06864632273813337, -2.1139672501544786, 0.0054126048112383505, 2.087273061794822, 1.843456595230639])

# Hand position in which we can grip the pen
pick_pos = np.array([0.2143865820313201, 0.4581297944284533, 0.08990505074553438])

up = np.array([0, 0, 1])


start_offset = 150/1000

# Move to intitial config
joint_move(start_q)

joint_move('Tool side start')
print('Align hand')
hand_align_pick_tool()
goto_xyz_hand(pick_pos + up*start_offset)
goto_xyz_hand(pick_pos)
grasp()
goto_xyz_hand(pick_pos + up*start_offset)
joint_move('Home')
