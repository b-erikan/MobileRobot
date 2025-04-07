#!/usr/bin/env python3
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

#Stellen des Arrays[Start_Init, Start_Joy, Start_MPC, Start_Pfad]
Startbit = np.array([0,0,0,0,0,0,0,0,0,0]) #Start_Array aus Hauptprogramm

#Stellen des Arrays[Pfad_Ziel_X, Pfad_Ziel_Y, Pfad_Ziel_Yaw, CSV_Nr, CSV_Nr_Pfad]
Daten_Hauptprogramm = np.array([0,0,0,0,0,0,0,0,0,0]) #Daten_Array aus Hauptprogramm

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

def joy_callback(data):
    #gas pedal
    #rt data is 1, when the trigger is release and -1 when pushed
    trigger_rt = -(data.axes[5]-1)/2 # rescaling to go from 0 to 1

    w_linear = 15*trigger_rt # rpm
    w_rotational = 15*trigger_rt #rpm
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

def get_start_joy(data):
    global Startbit

    Startbit=data.data

if __name__ == '__main__': 
    rospy.init_node('bigtoy_joycon_controller', anonymous=True)
    listener1=rospy.Subscriber('Start_Array',Float32MultiArray, get_start_joy, queue_size = 10)
    
    #while Startbit[1] == 0:
    #    print('Controller deaktiviert')
    #    rospy.sleep(2)
    
    listener2=rospy.Subscriber('joy', Joy, joy_callback, queue_size = 1)
   
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        r.sleep()
