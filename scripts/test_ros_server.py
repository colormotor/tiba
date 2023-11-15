#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

def msg_callback(data):
    rospy.loginfo(rospy.get_caller_id() + "Msg: %s", data.data)

def jtraj_callback(data):
    rospy.loginfo(rospy.get_caller_id() + "Joint trajectory with %d points"%(len(data.points)))
    Qs = [[p.positions for p in data.points]]
    dQs = [[p.velocities for p in data.points]]
    print(Qs)
    print(dQs)

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("/tiba/msg", String, msg_callback)
    rospy.Subscriber("/tiba/jtraj", JointTrajectory, jtraj_callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
