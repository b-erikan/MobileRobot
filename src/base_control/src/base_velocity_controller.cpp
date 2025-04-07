// BASE_VELOCITY_CONTROLLER.CPP
// minimal implementation of Canopen CiA 402 motor control for the faulhaber controller
// It subscribes the robot wheel velocity topic and sends corresponding CAN messages.

// simple keyboard control 
#include "mecaBox.h"
#include <iostream>
#include <fstream>
#include <string>
#include "ros/ros.h"
#include "std_msgs/String.h"
#include <base_control/base_wheel_vel.h>

float w_linear = 20; // wheel speed [rpm]
float w_rotational = 20;
double Omega_M[4];
mecaBox my_robot;

void wheelvelCallback(const base_control::base_wheel_vel::ConstPtr &msg){
    Omega_M[0]=msg->w1;
    Omega_M[1]=msg->w2;
    Omega_M[2]=msg->w3;
    Omega_M[3]=msg->w4;
    my_robot.wheel_velocity_command(Omega_M[0],Omega_M[1],Omega_M[2],Omega_M[3] );
    std::cout << "Velocity: " << Omega_M[0] << ", " << Omega_M[1] << ", "<< Omega_M[2] << ", "<< Omega_M[3] << std::endl;

}

int main(int argc, char **argv){
    std::cout << "Starting program..." << std::endl;
    ros::init(argc, argv, "base_motor_controller");
    std::cout << "ROS Node initialized." << std::endl;
    my_robot.initialize();
    std::cout << "CAN communication with Motor controllers established." << std::endl;

    ros::NodeHandle n;
    ros::Subscriber sub = n.subscribe("robot_wheel_command", 1000, wheelvelCallback);
    ros::spin();

    std::cout << "Closing program. Disabling motor voltage..." << std::endl;
    my_robot.stop();
    std::cout << "Program ends." << std::endl;
    return 0;
}