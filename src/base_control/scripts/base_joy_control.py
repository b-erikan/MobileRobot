#!/usr/bin/python3
# keyboard control for the farida_motor farida_epos node v0.4
# https://github.com/ros-teleop/teleop_twist_keyboard/blob/master/teleop_twist_keyboard.py


from __future__ import print_function
from email.errors import StartBoundaryNotFoundDefect

import threading

import roslib;
import rospy
import numpy as np

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from std_msgs.msg import *

import sys, select, termios, tty

from base_control.msg import *

msg = """
Reading from the teleop_twist_joy node and Publishing to Twist!
Subscribed to topic: /joy
---------------------------
Left Joystick: Linear movement
Right Joystick: Rotation
RT: Velocity
"""

pub = rospy.Publisher('robot_wheel_command', base_wheel_vel, queue_size = 10)
#pub = rospy.Publisher('robot_wheel_command_joy', base_wheel_vel, queue_size = 10)

joy_controller_on = False

def joy_callback(data: Joy):
    # print("joy received")
    wheel_speed_command = base_wheel_vel()

    global joy_controller_on

	# press A + RT to disable joystick control
    if data.buttons[0] and data.buttons[5]:
        joy_controller_on = False
	# press B + RT to enable joystick control
    elif data.buttons[1] and data.buttons[5]:
        joy_controller_on = True

    if joy_controller_on:
        #gas pedal
        #rt data is 1, when the trigger is release and -1 when pushed
        trigger_rt = -(data.axes[5]-1)/2 # rescaling to go from 0 to 1

        w_linear = 0.1*trigger_rt # m/s
        w_rotational = 0.1*trigger_rt # rad/s
        # direction: using definition of /joy topic
        joy_stick_left_longitudinal = data.axes[1] # positive: move forward
        joy_stick_left_lateral = data.axes[0] # positive: move left
        joy_stick_right_lateral = data.axes[3] # positive: rotate anti-clockwise

        vy = joy_stick_left_lateral * w_linear # x velocity
        vx = joy_stick_left_longitudinal * w_linear # y velocity
        w  = joy_stick_right_lateral * w_rotational # angular velocity

        r = 0.12 # radius of the wheels (m)
        lx = 0.64/2 # half distance between center of front wheels(m)
        ly = 0.62/2 # half distance between front and back wheels (m)

        # angular velocity (rad/s)
        w1 = (vx - vy - (lx + ly) * w)/r
        w2 = (vx + vy + (lx + ly) * w)/r
        w3 = (vx + vy - (lx + ly) * w)/r
        w4 = (vx - vy + (lx + ly) * w)/r

        # rpm
        wheel_speed_command.w1 = w1 * 60 / (2 * np.pi)
        wheel_speed_command.w2 = w2 * 60 / (2 * np.pi)
        wheel_speed_command.w3 = w3 * 60 / (2 * np.pi)
        wheel_speed_command.w4 = w4 * 60 / (2 * np.pi)
    else:
        return


    rospy.loginfo("sent command [%s, %s, %s, %s]", wheel_speed_command.w1, wheel_speed_command.w2, wheel_speed_command.w3, wheel_speed_command.w4)


    pub.publish(wheel_speed_command)


if __name__ == '__main__': 
    rospy.init_node('bigtoy_joycon_controller', anonymous=True)
    
    listener2=rospy.Subscriber('/joy', Joy, joy_callback, queue_size = 1)
   
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        r.sleep()
