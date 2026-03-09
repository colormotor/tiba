#!/usr/bin/env sh
#echo Setting ROS_MASTER_URI to https://horribly-next-shark.ngrok-free.app
export ROS_IPV6=off
#export ROS_HOSTNAME=158.223.47.153 #0.0.0.0
export ROS_MASTER_URI=http://172.16.0.1:11311 # IP of the Server
export ROS_IP=172.16.0.5 # On server: IP of server, On client: IP of client
export ROS_HOSTNAME=172.16.0.5 # On server: IP of server, On client: IP of client

