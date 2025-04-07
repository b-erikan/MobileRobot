#!/usr/bin/env python3
# keyboard control for the farida_motor farida_epos node v0.4
# https://github.com/ros-teleop/teleop_twist_keyboard/blob/master/teleop_twist_keyboard.py


from __future__ import print_function

import threading

import roslib;
import rospy
import numpy as np

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

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

def joy_callback(data):
    w_linear = 15 # rpm
    w_rotational = 15 #rpm
    # direction: using definition of /joy topic
    joy_stick_left_longitudinal = data.axes[1] # positive: move forward
    joy_stick_left_lateral = data.axes[0] # positive: move left
    joy_stick_right_lateral = data.axes[3] # positive: rotate anti-clockwise
    
    is_w_pressed = 0
    is_s_pressed = 0
    is_a_pressed = 0
    is_d_pressed = 0
    is_q_pressed = 0
    is_e_pressed = 0
    if joy_stick_left_longitudinal > 0:
        is_w_pressed = 1
    if joy_stick_left_longitudinal < 0:
        is_s_pressed = 1
    if joy_stick_left_lateral > 0:
        is_a_pressed = 1
    if joy_stick_left_lateral < 0:
        is_d_pressed = 1
    if joy_stick_right_lateral > 0:
        is_q_pressed = 1
    if joy_stick_right_lateral < 0:
        is_e_pressed = 1
    wheel_speed_command = base_wheel_vel()
    if is_w_pressed:
        wheel_speed_command.w1 = w_linear
        wheel_speed_command.w2 = w_linear
        wheel_speed_command.w3 = w_linear
        wheel_speed_command.w4 = w_linear
                
    if is_s_pressed:
        wheel_speed_command.w1 = -w_linear
        wheel_speed_command.w2 = -w_linear
        wheel_speed_command.w3 = -w_linear
        wheel_speed_command.w4 = -w_linear
    
    if is_a_pressed:
        wheel_speed_command.w1 = -w_linear
        wheel_speed_command.w2 = w_linear
        wheel_speed_command.w3 = w_linear
        wheel_speed_command.w4 = -w_linear

    if is_d_pressed:
        wheel_speed_command.w1 = w_linear
        wheel_speed_command.w2 = -w_linear
        wheel_speed_command.w3 = -w_linear
        wheel_speed_command.w4 = w_linear

    if is_q_pressed:
        wheel_speed_command.w1 = -w_rotational
        wheel_speed_command.w2 = w_rotational
        wheel_speed_command.w3 = -w_rotational
        wheel_speed_command.w4 = w_rotational

    if is_e_pressed:
        wheel_speed_command.w1 = w_rotational
        wheel_speed_command.w2 = -w_rotational
        wheel_speed_command.w3 = w_rotational
        wheel_speed_command.w4 = -w_rotational

    # linear + left rotation
    if is_w_pressed and is_q_pressed:
        wheel_speed_command.w1 = w_linear - w_rotational
        wheel_speed_command.w2 = w_linear + w_rotational
        wheel_speed_command.w3 = w_linear - w_rotational
        wheel_speed_command.w4 = w_linear + w_rotational
    if is_s_pressed and is_q_pressed:
        wheel_speed_command.w1 = -w_linear - w_rotational
        wheel_speed_command.w2 = -w_linear + w_rotational
        wheel_speed_command.w3 = -w_linear - w_rotational
        wheel_speed_command.w4 = -w_linear + w_rotational
    if is_a_pressed and is_q_pressed:
        wheel_speed_command.w1 = -w_linear - w_rotational
        wheel_speed_command.w2 = w_linear + w_rotational
        wheel_speed_command.w3 = w_linear - w_rotational
        wheel_speed_command.w4 = -w_linear + w_rotational
    if is_d_pressed and is_q_pressed:
        wheel_speed_command.w1 = w_linear - w_rotational
        wheel_speed_command.w2 = -w_linear + w_rotational
        wheel_speed_command.w3 = -w_linear - w_rotational
        wheel_speed_command.w4 = w_linear + w_rotational

    # linear + right rotation
    if is_w_pressed and is_e_pressed:
        wheel_speed_command.w1 = w_linear + w_rotational
        wheel_speed_command.w2 = w_linear - w_rotational
        wheel_speed_command.w3 = w_linear + w_rotational
        wheel_speed_command.w4 = w_linear - w_rotational
    if is_s_pressed and is_e_pressed:
        wheel_speed_command.w1 = -w_linear + w_rotational
        wheel_speed_command.w2 = -w_linear - w_rotational
        wheel_speed_command.w3 = -w_linear + w_rotational
        wheel_speed_command.w4 = -w_linear - w_rotational
    if is_a_pressed and is_q_pressed:
        wheel_speed_command.w1 = -w_linear + w_rotational
        wheel_speed_command.w2 = w_linear - w_rotational
        wheel_speed_command.w3 = w_linear + w_rotational
        wheel_speed_command.w4 = -w_linear - w_rotational
    if is_d_pressed and is_q_pressed:
        wheel_speed_command.w1 = w_linear + w_rotational
        wheel_speed_command.w2 = -w_linear - w_rotational
        wheel_speed_command.w3 = -w_linear + w_rotational
        wheel_speed_command.w4 = w_linear - w_rotational

    # linear + liear (45 deg)
    if is_w_pressed and is_a_pressed:
        wheel_speed_command.w1 = 0
        wheel_speed_command.w2 = w_linear
        wheel_speed_command.w3 = w_linear
        wheel_speed_command.w4 = 0
    if is_w_pressed and is_d_pressed:
        wheel_speed_command.w1 = w_linear*1.414
        wheel_speed_command.w2 = 0
        wheel_speed_command.w3 = 0
        wheel_speed_command.w4 = w_linear*1.414
    if is_s_pressed and is_a_pressed:
        wheel_speed_command.w1 = -w_linear*1.414
        wheel_speed_command.w2 = 0
        wheel_speed_command.w3 = 0
        wheel_speed_command.w4 = -w_linear*1.414
    if is_s_pressed and is_d_pressed:
        wheel_speed_command.w1 = 0
        wheel_speed_command.w2 = -w_linear*1.414
        wheel_speed_command.w3 = -w_linear*1.414
        wheel_speed_command.w4 = 0

    rospy.loginfo("sent command [%s, %s, %s, %s]", wheel_speed_command.w1, wheel_speed_command.w2, wheel_speed_command.w3, wheel_speed_command.w4)


    pub.publish(wheel_speed_command)


def joy_listener():
    print("Starting Joycon Command Node.")
    print("Left stick: linear movement.")
    print("Right stick: angular movement.")

    #start node
    rospy.init_node('bigtoy_joycon_controller', anonymous=True)
    rospy.Subscriber('joy', Joy, joy_callback, queue_size = 1)
    
    rospy.spin()

if __name__ == '__main__':
    try:
        joy_listener()
    except rospy.ROSInterruptException:
        pass
