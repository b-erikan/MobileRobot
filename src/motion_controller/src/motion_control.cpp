#include "motion_controller/motion_control.h"

// MPC Class Implementation
MPC::MPC()
{
    N_ = 10;
    dt_ = 0.01;
    u_max_x_ = 2;
    u_max_y_ = 2;
    w_max_ = 2;
    std::vector<double> weights = {100, 100, 100, 0, 0, 0, 1, 1, 1, 100, 100, 100, 0, 0, 0}; // Q,R,P
    u_min_x_ = -u_max_x_;
    u_min_y_ = -u_max_y_;
    w_min_ = -w_max_;

    Q_ = DM::zeros(6, 6);
    R_ = DM::zeros(3, 3);
    P_ = DM::zeros(6, 6);

    setWeights(weights);
    system_dynamics_ = setSystemModel();
}

MPC::~MPC() {}

void MPC::setWeights(std::vector<double> weights)
{
    Q_(0, 0) = weights[0];
    Q_(1, 1) = weights[1];
    Q_(2, 2) = weights[2];
    Q_(3, 3) = weights[3];
    Q_(4, 4) = weights[4];
    Q_(5, 5) = weights[5];

    R_(0, 0) = weights[6];
    R_(1, 1) = weights[7];
    R_(2, 2) = weights[8];

    P_(0, 0) = weights[9];
    P_(1, 1) = weights[10];
    P_(2, 2) = weights[11];
    P_(3, 3) = weights[12];
    P_(4, 4) = weights[13];
    P_(5, 5) = weights[14];
}

Function MPC::setSystemModel()
{
    // Symbolic variables for states
    MX p_x = MX::sym("p_x");
    MX p_y = MX::sym("p_y");
    MX theta = MX::sym("theta");
    MX v_x = MX::sym("v_x");
    MX v_y = MX::sym("v_y");
    MX w = MX::sym("w");

    MX state_vars = MX::vertcat({p_x, p_y, theta, v_x, v_y, w});

    MX v_x_ref = MX::sym("v_x_ref");
    MX v_y_ref = MX::sym("v_y_ref");
    MX w_ref = MX::sym("w_ref");
    MX control_vars = MX::vertcat({v_x_ref, v_y_ref, w_ref});

    const double tau_vx = 0.1;
    const double tau_vy = 0.1;
    const double tau_w = 0.2;

    MX R_z = MX::vertcat({MX::horzcat({MX::cos(theta), -MX::sin(theta), MX::zeros(1, 1)}),
                          MX::horzcat({MX::sin(theta), MX::cos(theta), MX::zeros(1, 1)}),
                          MX::horzcat({MX::zeros(1, 1), MX::zeros(1, 1), MX::ones(1, 1)})});
    MX G = dt_ * R_z;

    const double a1 = std::exp(-dt_ / tau_vx);
    const double a2 = std::exp(-dt_ / tau_vy);
    const double a3 = std::exp(-dt_ / tau_w);

    MX A_v = MX::diag(MX::vertcat({a1, a2, a3}));

    MX A = MX::vertcat({MX::horzcat({MX::eye(3), G}),
                        MX::horzcat({MX::zeros(3, 3), A_v})});

    const double b1 = (1 - std::exp(-dt_ / tau_vx));
    const double b2 = (1 - std::exp(-dt_ / tau_vy));
    const double b3 = (1 - std::exp(-dt_ / tau_w));

    MX B_v = MX::diag(MX::vertcat({b1, b2, b3}));

    MX B = MX::vertcat({MX::zeros(3, 3),
                        B_v});

    MX rhs = mtimes(A, state_vars) + mtimes(B, control_vars);

    return Function("system_dynamics", {state_vars, control_vars}, {rhs});
}

bool MPC::optimize(Eigen::VectorXd current_states, Eigen::MatrixXd desired_states)
{
    const int n_states = 6;
    const int n_controls = 3;
    Opti opti = Opti();

    Slice all;

    MX cost = 0;
    // 6 states, N+1 time steps
    X = opti.variable(n_states, N_ + 1);
    // 3 inputs, N time steps
    U = opti.variable(n_controls, N_);

    // Extract state variables
    MX p_x = X(0, all);
    MX p_y = X(1, all);
    MX theta = X(2, all);
    MX v_x = X(3, all);
    MX v_y = X(4, all);
    MX w = X(5, all);

    // Extract input variables
    MX v_x_ref = U(0, all);
    MX v_y_ref = U(1, all);
    MX w_ref = U(2, all);

    // Reference trajectory and current state
    MX X_ref = opti.parameter(6, N_ + 1);
    MX X_cur = opti.parameter(6);

    // Safely convert current states to CasADi matrix
    if (current_states.size() != 6)
    {
        ROS_ERROR("Current states vector must be exactly 6 elements long!");
        return false;
    }

    // Create a vector of doubles from Eigen vector
    std::vector<double> x_tmp_v(current_states.data(), current_states.data() + current_states.size());
    DM x_tmp1 = x_tmp_v;

    opti.set_value(X_cur, x_tmp1);

    // Convert desired states to CasADi matrix
    if (desired_states.rows() != 6 || desired_states.cols() != N_ + 1)
    {
        ROS_ERROR("Desired states matrix must be 6x(N+1)!");
        return false;
    }

    std::vector<double> X_ref_v(desired_states.data(),
                                desired_states.data() + desired_states.size());
    DM X_ref_d(X_ref_v);
    X_ref = MX::reshape(X_ref_d, n_states, N_ + 1);

    // Cost function
    for (int i = 0; i < N_; ++i)
    {
        MX X_err = X(all, i) - X_ref(all, i);
        MX U_0 = U(all, i);

        // State error cost
        cost += MX::mtimes({X_err.T(), Q_, X_err});

        // Control input cost
        cost += MX::mtimes({U_0.T(), R_, U_0});
    }

    // Terminal cost
    cost += MX::mtimes({(X(all, N_) - X_ref(all, N_)).T(), P_,
                        X(all, N_) - X_ref(all, N_)});

    opti.minimize(cost);

    // Dynamics constraints
    for (int i = 0; i < N_; ++i)
    {
        std::vector<MX> input(2);
        input[0] = X(all, i);
        input[1] = U(all, i);

        MX X_next = system_dynamics_(input)[0];
        opti.subject_to(X_next == X(all, i + 1));
    }

    // Initial state constraint
    opti.subject_to(X(all, 0) == X_cur);

    // Input constraints
    opti.subject_to(-u_max_x_ <= v_x_ref <= u_max_x_);
    opti.subject_to(-u_max_y_ <= v_y_ref <= u_max_y_);
    opti.subject_to(-w_max_ <= w_ref <= w_max_);

    // Solver configuration
    casadi::Dict solver_opts;
    solver_opts["expand"] = true;
    solver_opts["ipopt.max_iter"] = 10000;
    solver_opts["ipopt.print_level"] = 0;
    solver_opts["print_time"] = 0;
    solver_opts["ipopt.acceptable_tol"] = 1e-6;
    solver_opts["ipopt.acceptable_obj_change_tol"] = 1e-6;

    opti.solver("ipopt", solver_opts);

    // Solve the optimization problem
    solution_ = std::make_unique<casadi::OptiSol>(opti.solve());

    return true;
}

std::vector<double> MPC::getPredictU()
{
    std::vector<double> u_res;
    auto vel_cmd = solution_->value(U);

    for (int i = 0; i < N_; ++i)
    {
        u_res.push_back(static_cast<double>(vel_cmd(0, i)));
        u_res.push_back(static_cast<double>(vel_cmd(1, i)));
        u_res.push_back(static_cast<double>(vel_cmd(2, i)));
    }

    return u_res;
}

std::vector<double> MPC::getPredictX()
{
    std::vector<double> res;
    auto predict_x = solution_->value(X);

    for (int i = 0; i <= N_; ++i)
    {
        res.push_back(static_cast<double>(predict_x(0, i)));
        res.push_back(static_cast<double>(predict_x(1, i)));
        res.push_back(static_cast<double>(predict_x(2, i)));
        res.push_back(static_cast<double>(predict_x(3, i)));
        res.push_back(static_cast<double>(predict_x(4, i)));
        res.push_back(static_cast<double>(predict_x(5, i)));
    }
    return res;
}

int MPC::getHorizon()
{
    return N_;
}

// PID Class Implementation
PID::PID(float kp, float ki, float kd, float dt, float max_output_sqr)
    : kp_(kp), ki_(ki), kd_(kd), prev_error_(0.0), integral_(0.0), prev_output_(0.0), dt_(dt), max_output_sqr_(max_output_sqr) {}

float PID::update(float error)
{
    // TODO: ADD ANTI WIND-UP FOR INTEGRAL
    /*if (((prev_output_ > 0 && error > 0) || (prev_output_ < 0 && error < 0)) && (prev_output_ * prev_output_) >= max_output_sqr_)
    {
        integral_ = 0;
        ROS_ERROR("Integrator Wind-up!!");
    }
    else
    {
        integral_ += error;
    }*/

    integral_ += error * dt_;
    derivative_ = (error - prev_error_) / dt_;
    float output = kp_ * error + ki_ * integral_ + kd_ * derivative_;
    prev_error_ = error;
    prev_output_ = output;
    return output;
}

// Controller Class Implementation
Controller::Controller() : vx_meas(0.0), vy_meas(0.0), wz_meas(0.0), px_meas(0.0), py_meas(0.0), theta_meas(0.0)
{
    VelCmdPub = nh.advertise<base_control::base_wheel_vel>("robot_wheel_command", 10);
    PosFeedbackSub = nh.subscribe("/rtabmap/odom", 10,
                                  &Controller::stateEstimatorCallback, this);

    if (readWaypoints(waypoints, p_x, p_y, theta))
    {
        std::cout << "Successfully read " << p_x.size() << " rows from CSV file." << std::endl;
        printWaypoints(p_x, p_y, theta);
    }
}

void Controller::stateEstimatorCallback(const nav_msgs::Odometry::ConstPtr &msg)
{
    px_meas = msg->pose.pose.position.x;
    py_meas = msg->pose.pose.position.y;

    double w_tmp = msg->pose.pose.orientation.w;
    double x_tmp = msg->pose.pose.orientation.x;
    double y_tmp = msg->pose.pose.orientation.y;
    double z_tmp = msg->pose.pose.orientation.z;

    theta_meas = std::atan2(2 * (w_tmp * z_tmp + x_tmp * y_tmp), 1 - 2 * (y_tmp * y_tmp + z_tmp * z_tmp));

    vx_meas = msg->twist.twist.linear.x;
    vy_meas = msg->twist.twist.linear.y;
    wz_meas = msg->twist.twist.angular.z;

    ROS_INFO("Velocity -> V_x: %.2f, V_y: %.2f, w: %.2f", vx_meas, vy_meas, wz_meas);
}

void Controller::update_wheel_speeds(float vx, float vy, float wz)
{
    base_control::base_wheel_vel wheel_speed_command;

    /*w1:Front left
      w2:Front Right
      w3:Rear Left
      w4:Rear Right*/
    const float Lx = 0.65, Ly = 0.65, Rw = 0.125, w_max_sqr = 400;

    float w1 = (vx - vy - (Lx + Ly) * wz) / Rw;
    float w2 = (vx + vy + (Lx + Ly) * wz) / Rw;
    float w3 = (vx + vy - (Lx + Ly) * wz) / Rw;
    float w4 = (vx - vy + (Lx + Ly) * wz) / Rw;

    float max_wheel_speed_curr = fmax(fmax(fabs(w1), fabs(w2)), fmax(fabs(w3), fabs(w4)));

    // Scale if necessary
    if (max_wheel_speed_curr > max_wheel_speed)
    {
        ROS_ERROR("Wheel speed limit exceeded!!\n\n Front Left Wheel: %.2f\n Front Right Wheel: %.2f\n Rear Left Wheel:%.2f\n Rear Right Wheel: %.2f\n", w1, w2, w3, w4);
        float scale_factor = max_wheel_speed / max_wheel_speed_curr;
        w1 *= scale_factor;
        w2 *= scale_factor;
        w3 *= scale_factor;
        w4 *= scale_factor;
    }

    // Set wheel speeds
    wheel_speed_command.w1 = w1;
    wheel_speed_command.w2 = w2;
    wheel_speed_command.w3 = w3;
    wheel_speed_command.w4 = w4;

    // Publish the wheel speeds
    VelCmdPub.publish(wheel_speed_command);
}

void Controller::startControlLoop()
{
    PID PID_Vx(1, 0.1, 0.05, dt, 2);
    PID PID_Vy(0.01, 0.01, 0.01, dt, 2);
    PID PID_Wz(3, 1.2, 0, dt, 3);
    trajectory_planner = std::make_unique<MPC>();

    // Define a target state (desired trajectory)
    Eigen::MatrixXd desired_states(6, trajectory_planner->getHorizon() + 1);
    for (int i = 0; i <= trajectory_planner->getHorizon(); ++i)
    {
        desired_states(0, i) = 1.5;
        desired_states(1, i) = 0.0;
        desired_states(2, i) = 3.14;
        desired_states(3, i) = 0.0;
        desired_states(4, i) = 0.0;
        desired_states(5, i) = 0.0;
    }

    ros::Rate rate(1 / dt);
    while (ros::ok())
    {
        // float vx_des = 0.0, vy_des = 0, wz_des = 0.5;
        //  CAUTION!! The position measurements are given in the map (fixed) frame. The velocity measurements are given in the body frame!!!!
        Eigen::VectorXd current_state(6);
        current_state << px_meas, py_meas, theta_meas, vx_meas, vy_meas, wz_meas;
        // current_state << 0, 0, 0, 0, 0, 0;
        std::vector<double> predicted_x;
        std::vector<double> predicted_u;

        trajectory_planner->optimize(current_state, desired_states);
        predicted_x = trajectory_planner->getPredictX();
        predicted_u = trajectory_planner->getPredictU(); // Get the first control inputs

        /*
        // Print header
        std::cout << "\n===== MPC Predictions =====" << std::endl;

        // Print Control Inputs (U)
        std::cout << "\nPredicted Control Inputs (U):" << std::endl;
        std::cout << "Total U entries: " << predicted_x.size() << std::endl;
        for (size_t i = 0; i < predicted_u.size(); i += 3)
        {
            std::cout << "Time Step " << i / 3 << ": "
                      << "v_x_ref = " << predicted_u[i]
                      << ", v_y_ref = " << predicted_u[i + 1]
                      << ", w_ref = " << predicted_u[i + 2] << std::endl;
        }

        // Print States (X)
        std::cout << "\nPredicted States (X):" << std::endl;
        std::cout << "Total X entries: " << predicted_x.size() << std::endl;
        for (size_t i = 0; i < predicted_x.size(); i += 6)
        {
            std::cout << "Time Step " << i / 6 << ": "
                      << "x = " << predicted_x[i]
                      << ", y = " << predicted_x[i + 1]
                      << ", theta = " << predicted_x[i + 2]
                      << ", v_x = " << predicted_x[i + 3]
                      << ", v_y = " << predicted_x[i + 4]
                      << ", w = " << predicted_x[i + 5] << std::endl;
        }

        std::cout << "\n===== End of Predictions =====" << std::endl;
        */

        /*
        // Calculate errors

        volatile float vx_err = - vx_meas;
        volatile float vy_err = 0 - vy_meas;
         volatile float wz_err = 0 - wz_meas;

         //Update control effort
         float cmd_Vx = PID_Vx.update(vx_err);
         float cmd_Vy = PID_Vy.update(vy_err);
         float cmd_Wz = PID_Wz.update(wz_err);

         //Execute control
         update_wheel_speeds(cmd_Vx, cmd_Vy, cmd_Wz);
         ROS_INFO("V_x: %.2f, V_y: %.2f W_z: %.2f", vx_meas, vy_meas, wz_meas);
         ROS_INFO("E_x: %.2f, E_y: %.2f E_z: %.2f", vx_err, vy_err, wz_err);
         ROS_INFO("Cmd_x: %.2f, Cmd_y: %.2f Cmd_z: %.2f", cmd_Vx, cmd_Vy, cmd_Wz);
        //update_wheel_speeds(predicted_u[0], predicted_u[1], predicted_u[2]);
        //ROS_INFO("V_x: %.2f, V_y: %.2f W_z: %.2f", vx_meas, vy_meas, wz_meas);
        //ROS_INFO("P_x: %.2f, P_y: %.2f theta: %.2f", px_meas, py_meas, theta_meas);
        //ROS_INFO("Cmd_x: %.2f, Cmd_y: %.2f Cmd_z: %.2f", predicted_u[0], predicted_u[1], predicted_u[2]);*/
        ros::spinOnce();
        rate.sleep();
    }
}
bool Controller::readWaypoints(const std::string &filePath,
                               std::vector<double> &p_x,
                               std::vector<double> &p_y,
                               std::vector<double> &theta)
{

    // Clear vectors in case they contain data
    p_x.clear();
    p_y.clear();
    theta.clear();

    // Open file
    std::ifstream file(filePath);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file '" << filePath << "'" << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string cell;
        double val1, val2, val3;

        // Read first column
        if (!std::getline(ss, cell, ','))
        {
            std::cerr << "Error: Failed to read X position!" << std::endl;
            continue;
        }
        try
        {
            val1 = std::stod(cell);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error converting '" << cell << "' to double: " << e.what() << std::endl;
            continue;
        }

        // Read second column
        if (!std::getline(ss, cell, ','))
        {
            std::cerr << "Error: Failed to read Y position!" << std::endl;
            continue;
        }
        try
        {
            val2 = std::stod(cell);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error converting '" << cell << "' to double: " << e.what() << std::endl;
            continue;
        }

        // Read third column
        if (!std::getline(ss, cell))
        {
            std::cerr << "Error: Failed to read the rotation angle!" << std::endl;
            continue;
        }
        try
        {
            val3 = std::stod(cell);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error converting '" << cell << "' to double: " << e.what() << std::endl;
            continue;
        }

        // Store values in respective vectors
        p_x.push_back(val1);
        p_y.push_back(val2);
        theta.push_back(val3);
    }

    file.close();
    return true;
}

void Controller::printWaypoints(const std::vector<double> &p_x,
                                const std::vector<double> &p_y,
                                const std::vector<double> &theta)
{

    // Check if vectors have the same size
    if (p_x.size() != p_y.size() || p_x.size() != theta.size())
    {
        std::cerr << "Error: Vectors have different sizes!" << std::endl;
        std::cerr << "p_x size: " << p_x.size() << ", p_y size: " << p_y.size()
                  << ", theta size: " << theta.size() << std::endl;
        return;
    }

    // Print header
    std::cout << std::string(50, '-') << std::endl;
    std::cout << std::setw(6) << "Index" << " | "
              << std::setw(15) << "p_x (m)" << " | "
              << std::setw(15) << "p_y (m)" << " | "
              << std::setw(15) << "theta (rad)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    // Set floating point precision for output
    std::cout << std::fixed << std::setprecision(4);

    // Print data rows
    for (size_t i = 0; i < p_x.size(); ++i)
    {
        std::cout << std::setw(6) << i << " | "
                  << std::setw(15) << p_x[i] << " | "
                  << std::setw(15) << p_y[i] << " | "
                  << std::setw(15) << theta[i] << std::endl;
    }

    // Print footer
    std::cout << std::string(50, '-') << std::endl;
    std::cout << "Total entries: " << p_x.size() << std::endl;
}

std::vector<TrajectoryPoint> Controller::generateTrajectory(const std::vector<Point> &waypoints, double linearSpeed)
{
    if (waypoints.size() < 2)
    {
        std::cerr << "Error: At least two waypoints are required for trajectory generation." << std::endl;
        return {};
    }

    std::vector<TrajectoryPoint> trajectory;

    // Add the first waypoint at time 0
    trajectory.push_back({waypoints[0].x, waypoints[0].y, waypoints[0].theta, 0.0});

    double cumulativeTime = 0.0;
    double totalDistance = 0.0;

    // Calculate distances between consecutive waypoints and add trajectory points
    for (size_t i = 1; i < waypoints.size(); ++i)
    {
        // Calculate Euclidean distance between consecutive points
        double dx = waypoints[i].x - waypoints[i - 1].x;
        double dy = waypoints[i].y - waypoints[i - 1].y;
        double segmentDistance = std::sqrt(dx * dx + dy * dy);

        // Calculate time needed to travel this segment at constant speed
        double segmentTime = segmentDistance / linearSpeed;
        cumulativeTime += segmentTime;
        totalDistance += segmentDistance;

        // Create trajectory point with timing information
        trajectory.push_back({waypoints[i].x,
                              waypoints[i].y,
                              waypoints[i].theta,
                              cumulativeTime});
    }

    std::cout << "Trajectory generated with " << trajectory.size() << " points." << std::endl;
    std::cout << "Total distance: " << totalDistance << " units" << std::endl;
    std::cout << "Total time: " << cumulativeTime << " seconds" << std::endl;

    return trajectory;
}

/**
 * Interpolates the trajectory to generate points at a fixed time interval
 * Useful for controllers that need regular sampling
 *
 * @param trajectory Original trajectory with variable time intervals
 * @param timeStep Desired fixed time step (seconds)
 * @return Trajectory with points at regular time intervals
 */
std::vector<TrajectoryPoint> Controller::interpolateTrajectory(
    const std::vector<TrajectoryPoint> &trajectory,
    double timeStep)
{
    if (trajectory.empty())
    {
        return {};
    }

    std::vector<TrajectoryPoint> interpolatedTrajectory;

    // Start time
    double startTime = trajectory.front().time;
    // End time
    double endTime = trajectory.back().time;

    // Generate points at regular intervals
    for (double t = startTime; t <= endTime; t += timeStep)
    {
        // Find the two points to interpolate between
        auto it = std::lower_bound(
            trajectory.begin(),
            trajectory.end(),
            t,
            [](const TrajectoryPoint &point, double time)
            {
                return point.time < time;
            });

        // Handle edge cases
        if (it == trajectory.begin())
        {
            interpolatedTrajectory.push_back(trajectory.front());
            continue;
        }

        if (it == trajectory.end())
        {
            interpolatedTrajectory.push_back(trajectory.back());
            continue;
        }

        // Get the two points for interpolation
        const TrajectoryPoint &p2 = *it;
        const TrajectoryPoint &p1 = *(it - 1);

        // Calculate interpolation factor
        double alpha = (t - p1.time) / (p2.time - p1.time);

        // Linear interpolation
        TrajectoryPoint interpolated;
        interpolated.x = p1.x + alpha * (p2.x - p1.x);
        interpolated.y = p1.y + alpha * (p2.y - p1.y);

        // Angular interpolation (be careful with angle wrapping)
        double angleDiff = p2.theta - p1.theta;
        // Normalize to [-pi, pi]
        if (angleDiff > M_PI)
            angleDiff -= 2 * M_PI;
        if (angleDiff < -M_PI)
            angleDiff += 2 * M_PI;

        interpolated.theta = p1.theta + alpha * angleDiff;
        interpolated.time = t;

        interpolatedTrajectory.push_back(interpolated);
    }

    return interpolatedTrajectory;
}

void Controller::writeTrajectoryToCSV(const std::vector<TrajectoryPoint> &trajectory, const std::string &filename)
{
    std::ofstream file("/home/prox/CSV_Listen/Ahmet.csv");
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file '" << "/home/prox/CSV_Listen/Ahmet.csv" << "' for writing." << std::endl;
        return;
    }

    // Write header
    file << "time,x,y,theta\n";

    // Write data
    for (const auto &point : trajectory)
    {
        file << point.time << ","
             << point.x << ","
             << point.y << ","
             << point.theta << "\n";
    }

    file.close();
    std::cout << "Trajectory written to " << "/home/prox/CSV_Listen/Ahmet.csv" << std::endl;
}


// Main Function
int main(int argc, char **argv)
{
    ros::init(argc, argv, "motion_control");

    Controller controller;
    controller.startControlLoop();

    return 0;
}

