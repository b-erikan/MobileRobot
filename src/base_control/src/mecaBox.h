/* youBox version 1.2021.09     */
/* By M.Sc. Xiang Chen          */
/* Insitute of Control Systems  */
/* University of Kaiserslautern */
/* chen@eit.uni-kl.de         */

#ifndef MECABOX_H
#define MECABOX_H

#include "time.h"
#include <vector>
#include <cmath>
#include <cstdint>
#include "SocketNTCAN.h"

/*** BEGIN: HELP FUNCTIONS ***/
struct timespec timespec_add(struct timespec time1, struct timespec time2);
CMSG int2hexbyte(int velocity);
/*** END: HELP FUNCTIONS ***/
const int VELOCITY_FACTOR = 4096; // 4096 vunit = 1rpm


class mecaBox
{
private:
    struct wheel_sensor_data
    {
        /* index 0 : front left  */
        /*       1 : front right */
        /*       2 : back left   */
        /*       3 : back right  */
        std::vector<std::int32_t> wheel_motor_encoder_tick_measured;
        std::vector<std::int32_t> wheel_motor_velocity_RPM_measured;
        std::vector<std::int32_t> wheel_motor_current_mA_measured;
        std::vector<double> wheel_position_measured;
        std::vector<double> wheel_velocity_measured;
        std::vector<double> wheel_torque_measured;

        wheel_sensor_data() : wheel_motor_encoder_tick_measured(4), wheel_motor_velocity_RPM_measured(4), wheel_motor_current_mA_measured(4), wheel_position_measured(4), wheel_velocity_measured(4), wheel_torque_measured(4) {}
    };

    struct arm_sensor_data arm_all_read(std::int8_t init);
    struct wheel_sensor_data wheel_all_read(std::int8_t init);
    std::int8_t wheel_position_command_err(double q1, double q2, double q3, double q4);
    std::int8_t wheel_velocity_command_err(double w1, double w2, double w3, double w4);
    std::int8_t wheel_torque_command_err(double t1, double t2, double t3, double t4);
    std::int8_t danger_detection(void);
    

public:
    int status(void);
    int initialize(void);
    int stop(void);
    int wheel_position_command(double q1, double q2, double q3, double q4);
    int wheel_velocity_command(double w1, double w2, double w3, double w4); // unit: rpm
    int wheel_torque_command(double t1, double t2, double t3, double t4);
    int wheel_position_read(std::vector<double> &position_rad);
    int wheel_velocity_read(std::vector<double> &velocity_rad_sec);
    int wheel_torque_read(std::vector<double> &torque_Nm);
};

#endif // MECABOX_H
