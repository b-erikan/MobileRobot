#!/usr/bin/env python3
# 

import rospy
from base_control.msg import *
import os
import sys
import tty, termios

from pynput import keyboard
from pynput.keyboard import Key, Listener

is_w_pressed = False
is_s_pressed = False
is_a_pressed = False
is_d_pressed = False
is_q_pressed = False
is_e_pressed = False

def on_press(key):
    #print(key.char)
    global is_w_pressed 
    global is_s_pressed 
    global is_a_pressed 
    global is_d_pressed 
    global is_q_pressed
    global is_e_pressed
    try:
        #print('Key presed: {0} '. format(key.char))
        if key.char == 'w':
            is_w_pressed = True
        elif key.char == 's':
            is_s_pressed = True
        elif key.char == 'a':
            is_a_pressed = True
        elif key.char == 'd':
            is_d_pressed = True
        elif key.char == 'q':
            is_q_pressed = True
        elif key.char == 'e':
            is_e_pressed = True
        print("w: " + str(is_w_pressed) + ", s: " + str(is_s_pressed) + ", a: " +str(is_a_pressed) + ", d: " +str(is_d_pressed) + ", q: " + str(is_q_pressed) + ", e: " +str(is_e_pressed))
    except AttributeError:
        print('special key pressed: {0}'.format(key))

def on_release(key):
    global is_w_pressed 
    global is_s_pressed 
    global is_a_pressed 
    global is_d_pressed 
    global is_q_pressed
    global is_e_pressed
    #print('Key released: {0}'.format(key))
    if key.char == 'w':
        is_w_pressed = False
    elif key.char == 's':
        is_s_pressed = False
    elif key.char == 'a':
        is_a_pressed = False
    elif key.char == 'd':
        is_d_pressed = False
    elif key.char == 'q':
        is_q_pressed = False
    elif key.char == 'e':
        is_e_pressed = False
    print("w: " + str(is_w_pressed) + ", s: " + str(is_s_pressed) + ", a: " +str(is_a_pressed) + ", d: " +str(is_d_pressed) + ", q: " + str(is_q_pressed) + ", e: " +str(is_e_pressed))
    
    if key == keyboard.Key.esc:
        # Stop listener
        return False

def velocity_publisher():
    global is_w_pressed 
    global is_s_pressed 
    global is_a_pressed 
    global is_d_pressed 
    global is_q_pressed
    global is_e_pressed
    pub = rospy.Publisher('robot_wheel_command', base_wheel_vel, queue_size = 10)
    rospy.init_node('keyboard_publisher', anonymous=True)

    # setup keyboard listener 
    listener = Listener(on_press=on_press, on_release=on_release, suppress=False)
    listener.start()
    
    print("Starting Keyboard Command Node.")
    print("Use WASD and QE to navigate. Press 'p' to end.")
    rate=rospy.Rate(10) #Hz
    wheel_speed_command = base_wheel_vel()
    q = 1
    w_linear = 20
    w_rotational = 20
    while not rospy.is_shutdown():
        wheel_speed_command.w1 = 0
        wheel_speed_command.w2 = 0
        wheel_speed_command.w3 = 0
        wheel_speed_command.w4 = 0
        # single dof motion       
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

        pub.publish(wheel_speed_command)
        rate.sleep()
        #print(wheel_speed_command)
    print("Node end.")

if __name__=='__main__':
    try:
        velocity_publisher()
    except rospy.ROSInterruptException:
        pass