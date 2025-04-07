#ifndef MPC_CONTROLLER_H
#define MPC_CONTROLLER_H

#include <iostream>
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Int32.h>
#include <cmath>
#include <base_control/base_wheel_vel.h>
#include <casadi/casadi.hpp>
#include <Eigen/Core>
#include <vector>
#include <memory>
#include <string>
#include <sstream>
#include <algorithm>
#include <fstream>  


using namespace casadi;


struct Point {
    double x;
    double y;
    double theta;  // orientation in radians
};

struct TrajectoryPoint {
    double x;
    double y;
    double theta;
    double time;   // time to reach this point (seconds from start)
};


class MPC
{
public:
    MPC();
    ~MPC();
    Function setSystemModel();
    void setWeights(std::vector<double> weights);
    //bool solve(Eigen::Vector3d current_states, Eigen::MatrixXd desired_states);
    bool optimize(Eigen::VectorXd current_states, Eigen::MatrixXd desired_states);
    std::vector<double> getPredictU();
    std::vector<double> getPredictX();
    int getHorizon();

private:
    int N_;     
    double dt_; 
    double u_max_x_, u_max_y_, w_max_, u_min_x_, u_min_y_, w_min_;

    // weights
    DM Q_, R_, P_;

    MX X;
    MX U;

    Function kinematic_equation_;
    Function system_dynamics_;
    std::unique_ptr<casadi::OptiSol> solution_;
};

class PID
{
private:
    float kp_, ki_, kd_, max_output_sqr_;
    float prev_error_, integral_, prev_output_;
    float derivative_;
    float dt_; // Time step

public:
    PID(float kp, float ki, float kd, float dt, float max_output_sqr);
    float update(float error);
};

class Controller
{
private:
    ros::NodeHandle nh;
    ros::Subscriber PosFeedbackSub;
    ros::Publisher VelCmdPub;
    Eigen::VectorXd current_state;
    std::unique_ptr<MPC> trajectory_planner;
    std::string waypoints = "/home/prox/CSV_Listen/Ahmet.csv";
    std::vector<double> p_x, p_y, theta;

public:
    volatile float vx_meas, vy_meas, wz_meas, px_meas, py_meas, theta_meas;
    const float dt = 1.f / 100.f;
    const float max_wheel_speed=50.0;
    
    Controller();
    void stateEstimatorCallback(const nav_msgs::Odometry::ConstPtr &msg);
    void update_wheel_speeds(float vx, float vy, float wz);
    void startControlLoop();
    bool readWaypoints(const std::string& filePath,std::vector<double>& p_x,std::vector<double>& p_y,std::vector<double>& theta);
    void printWaypoints(const std::vector<double>& p_x,const std::vector<double>& p_y,const std::vector<double>& theta);
    void writeTrajectoryToCSV(const std::vector<TrajectoryPoint> &trajectory, const std::string &filename);
    std::vector<TrajectoryPoint> interpolateTrajectory(const std::vector<TrajectoryPoint> &trajectory,double timeStep);
    std::vector<TrajectoryPoint> generateTrajectory(const std::vector<Point> &waypoints, double linearSpeed);
};



#endif // MPC_CONTROLLER_H