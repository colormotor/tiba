# History

The Draw-lab robotics lab in Goldsmiths has been recently togther with the EACVA project. The lab features a collaborative robot arm equipped with vision feedback and a pen-plotter. Both platforms can be easily adapted with different kinds of drawing tools. While less mature than the Konstanz lab, the Goldsmiths lab is under rapid development and provides additional flexibility when it comes to the implementation of safe artist-robot collaboration scenarios.

# Available painting machines
## Franka Emika Research 3 (FR3) 
The FR3 is a seven-axis collaborative robot (cobot) that is widely used within the robotic research community. Its seven degrees of freedom (DOF) result in a degree of articulation similar to a human arm, a feature that is particularly useful when controlling the orientation of a drawing tool. The FR3 is designed to enable full customization of its control loop, which enables the implementation of controllers that can react to perturbations with varying levels of stiffness. The robot has two "guiding" buttons on the wrist that allow a user to rapidly demonstrate and record motions that can be replayed by the robot.

### Drawing tool attachment
Drawing tools can be easily used by the robot through a custom built 3d printed attachment that connects to a screw based pen-holder identical to the one used by the AxiDraw plotter. The attachment is designed so it fits tightly in the gripper of the robot, which enables simple switching between different tools.

### Drawing pad
The robotic arm is mounted on a work-bench in front of a custom built drawing pad with adjustale inclination. The drawing pad is constructed with aluminium profiles and can be easily extended with pen/brush holders and palettes. 
The guiding-mode of the robot can be used to rapidly create new motions for picking/placing drawing tools or to reach palettes and ink/paint containers.

### Vision system
The robot is equipped with a wrist-mounted Intel Realsense D435 color and depth camera. We also provide a higher resolution Canon EOS 2000D camera, which can also be controlled programmatically. 

### Software
The control software is currently under heavy development. It currently enables recording and replaying recording demonstrated motions such as picking/placing, calibration of the drawing plane, path planning, computing the inverse kinematics for drawing tool trajectories, visualizing the resulting joint trajectories, simulating the behavior of the robot and finally executing the motions with the robot. The software is mostly written in Python with a C++ backend and is easy to extend and customize, while taking adavntage of the rich Python library echosystem. The software can also be easily interfaced with the Robotics Operating System (ROS) but currently does not require it to run.

## AxiDraw SE A3
The Draw-lab also provides access to an AxiDraw SE A3 pen plotter. A popular choice among digital/generative artists, the AxiDraw is an easy to use pen plotter that can be easily equipped with a variety of drawing tools and easily controlled through the vector drawing program InkScape or through a Python programming interface.

