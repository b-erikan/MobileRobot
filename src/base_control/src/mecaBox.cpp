/* youBox version 1.2021.09     */
/* By M.Sc. Xiang Chen          */
/* Insitute of Control Systems  */
/* University of Kaiserslautern */
/* chen@eit.uni-kl.de         */


#include <unistd.h>
#include <iostream>
#include <thread>
#include <mutex>
#include "SocketNTCAN.h"
#include "mecaBox.h"

/**** BEGIN: limits ****/
#define WHEEL_CURRENT_CMD_LIMIT 3000 //mA
#define WHEEL_VELOCITY_CMD_LIMIT 20 //rpm

#define WHEEL_CURRENT_LIMIT 3000 //mA
#define WHEEL_VELOCITY_LIMIT 20 //rpm
/**** END: limits ****/

/**** BEGIN: some constants ****/
const int ONE_SEC = 1000000;
const int ONE_MS = 1000;
/**** END: some constants ****/

void int2hexbytes(CMSG &cmsg, int velocity){
    for(int i = 4; i< 8; i++){
        cmsg.data[i] = velocity;
        velocity = velocity >> 8;
    }
}

int mecaBox::status(void){
    // return status of CAN connection
    return 0;
}

SocketNTCAN *canmaster= new SocketNTCAN();

int mecaBox::initialize(void){
    // initialize CAN connection and set wheels to Profile Velocity Mode
    canmaster->open();
    std::cout << "[mecaBox] CAN bus initialized." << std::endl;
    usleep(ONE_SEC);
    // start remote nodes
    canmaster->cmsg_tx[0].id = 0x000;
    canmaster->cmsg_tx[0].len = 0x02;
    canmaster->cmsg_tx[0].len |= canmaster->cmsg_tx[0].len + (canmaster->rtr<<4);
    canmaster->cmsg_tx[0].data[0] = 0x01;
    canmaster->cmsg_tx[0].data[1] = 0x00;
    canmaster->transmit();
    std::cout << "[mecaBox] NMT: start remote nodes." << std::endl;
    usleep(100*ONE_MS);
    // ready to switch on
    canmaster->cmsg_tx[0].id = 0x601;
    canmaster->cmsg_tx[0].len = 0x08;
    canmaster->cmsg_tx[0].len |= canmaster->cmsg_tx[0].len + (canmaster->rtr<<4);
    canmaster->cmsg_tx[0].data[0] = 0x2B;
    canmaster->cmsg_tx[0].data[1] = 0x40;
    canmaster->cmsg_tx[0].data[2] = 0x60;
    canmaster->cmsg_tx[0].data[3] = 0x00;
    canmaster->cmsg_tx[0].data[4] = 0x06;
    canmaster->cmsg_tx[0].data[5] = 0x00;
    canmaster->cmsg_tx[0].data[6] = 0x00;
    canmaster->cmsg_tx[0].data[7] = 0x00;
    canmaster->transmit();
    canmaster->cmsg_tx[0].id = 0x602;
    canmaster->transmit();
    canmaster->cmsg_tx[0].id = 0x603;
    canmaster->transmit();  
    canmaster->cmsg_tx[0].id = 0x604;
    canmaster->transmit(); 
    std::cout << "[mecaBox] SDO: Ready to switch on" << std::endl;
    usleep(100*ONE_MS);
    // switch on
    canmaster->cmsg_tx[0].id = 0x601;
    canmaster->cmsg_tx[0].len = 0x08;
    canmaster->cmsg_tx[0].len |= canmaster->cmsg_tx[0].len + (canmaster->rtr<<4);
    canmaster->cmsg_tx[0].data[0] = 0x2B;
    canmaster->cmsg_tx[0].data[1] = 0x40;
    canmaster->cmsg_tx[0].data[2] = 0x60;
    canmaster->cmsg_tx[0].data[3] = 0x00;
    canmaster->cmsg_tx[0].data[4] = 0x06;
    canmaster->cmsg_tx[0].data[5] = 0x00;
    canmaster->cmsg_tx[0].data[6] = 0x00;
    canmaster->cmsg_tx[0].data[7] = 0x00;
    canmaster->transmit();
    canmaster->cmsg_tx[0].id = 0x602;
    canmaster->transmit();
    canmaster->cmsg_tx[0].id = 0x603;
    canmaster->transmit();  
    canmaster->cmsg_tx[0].id = 0x604;
    canmaster->transmit(); 
    std::cout << "[mecaBox] SDO: switch on" << std::endl;
    usleep(100*ONE_MS);
    // enable operation
    canmaster->cmsg_tx[0].id = 0x601;
    canmaster->cmsg_tx[0].len = 0x08;
    canmaster->cmsg_tx[0].len |= canmaster->cmsg_tx[0].len + (canmaster->rtr<<4);
    canmaster->cmsg_tx[0].data[0] = 0x2B;
    canmaster->cmsg_tx[0].data[1] = 0x40;
    canmaster->cmsg_tx[0].data[2] = 0x60;
    canmaster->cmsg_tx[0].data[3] = 0x00;
    canmaster->cmsg_tx[0].data[4] = 0x0F;
    canmaster->cmsg_tx[0].data[5] = 0x00;
    canmaster->cmsg_tx[0].data[6] = 0x00;
    canmaster->cmsg_tx[0].data[7] = 0x00;
    canmaster->transmit();
    canmaster->cmsg_tx[0].id = 0x602;
    canmaster->transmit();
    canmaster->cmsg_tx[0].id = 0x603;
    canmaster->transmit();  
    canmaster->cmsg_tx[0].id = 0x604;
    canmaster->transmit(); 
    std::cout << "[mecaBox] SDO: enable operation" << std::endl;
    usleep(100*ONE_MS);

    // set profile velocity mode
    canmaster->cmsg_tx[0].id = 0x601;
    canmaster->cmsg_tx[0].len = 0x08;
    canmaster->cmsg_tx[0].len |= canmaster->cmsg_tx[0].len + (canmaster->rtr<<4);
    canmaster->cmsg_tx[0].data[0] = 0x2F;
    canmaster->cmsg_tx[0].data[1] = 0x60;
    canmaster->cmsg_tx[0].data[2] = 0x60;
    canmaster->cmsg_tx[0].data[3] = 0x00;
    canmaster->cmsg_tx[0].data[4] = 0x03;
    canmaster->cmsg_tx[0].data[5] = 0x00;
    canmaster->cmsg_tx[0].data[6] = 0x00;
    canmaster->cmsg_tx[0].data[7] = 0x00;
    canmaster->transmit();
    canmaster->cmsg_tx[0].id = 0x602;
    canmaster->transmit();
    canmaster->cmsg_tx[0].id = 0x603;
    canmaster->transmit();  
    canmaster->cmsg_tx[0].id = 0x604;
    canmaster->transmit(); 
    usleep(100*ONE_MS);
    std::cout << "[mecaBox] SDO: Profile velocity mode activated. Ready for operation." << std::endl;
    std::cout << "[mecaBOX] Initialization Done." << std::endl;
    return 0;
}

int mecaBox::stop(void){
    std::cout << "[mecaBox] Stopping control..." << std::endl;
    // disable voltage
    canmaster->cmsg_tx[0].id = 0x601;
    canmaster->cmsg_tx[0].len = 0x08;
    canmaster->cmsg_tx[0].len |= canmaster->cmsg_tx[0].len + (canmaster->rtr<<4);
    canmaster->cmsg_tx[0].data[0] = 0x2B;
    canmaster->cmsg_tx[0].data[1] = 0x40;
    canmaster->cmsg_tx[0].data[2] = 0x60;
    canmaster->cmsg_tx[0].data[3] = 0x00;
    canmaster->cmsg_tx[0].data[4] = 0x0D;
    canmaster->cmsg_tx[0].data[5] = 0x00;
    canmaster->cmsg_tx[0].data[6] = 0x00;
    canmaster->cmsg_tx[0].data[7] = 0x00;
    canmaster->transmit();
    canmaster->cmsg_tx[0].id = 0x602;
    canmaster->transmit();
    canmaster->cmsg_tx[0].id = 0x603;
    canmaster->transmit();  
    canmaster->cmsg_tx[0].id = 0x604;
    canmaster->transmit(); 
    std::cout << "[mecaBox] Voltage disabled." << std::endl;
    usleep(100*ONE_MS);
    canmaster->close();
    std::cout << "[mecaBox] Close CAN Port." << std::endl;
} 

int mecaBox::wheel_velocity_command(double w1, double w2, double w3, double w4)
{
    /* robot wheel velocity control, the inputs in rpm */
    /* w1: front left wheel;     w2: front right wheel;    */
    /* w3: back left wheel;      w4: back right wheel;     */
    // unit of input: rpm of wheel side
    // also note that velocity of w1 and w3 should be inverted, because the positive rotation direction is clockwise.

    int w1_int = -w1*VELOCITY_FACTOR;
    int w2_int = w2*VELOCITY_FACTOR;
    int w3_int = -w3*VELOCITY_FACTOR;
    int w4_int = w4*VELOCITY_FACTOR; 

    // set target velocity
    canmaster->cmsg_tx[0].id = 0x601;
    canmaster->cmsg_tx[0].len = 0x08;
    canmaster->cmsg_tx[0].len |= canmaster->cmsg_tx[0].len + (canmaster->rtr<<4);
    canmaster->cmsg_tx[0].data[0] = 0x23;
    canmaster->cmsg_tx[0].data[1] = 0xFF;
    canmaster->cmsg_tx[0].data[2] = 0x60;
    canmaster->cmsg_tx[0].data[3] = 0x00;
    int2hexbytes(canmaster->cmsg_tx[0], w1_int);
    canmaster->transmit();
    canmaster->cmsg_tx[0].id = 0x602;
    int2hexbytes(canmaster->cmsg_tx[0], w2_int);
    canmaster->transmit();
    canmaster->cmsg_tx[0].id = 0x603;
    int2hexbytes(canmaster->cmsg_tx[0], w3_int);
    canmaster->transmit();
    canmaster->cmsg_tx[0].id = 0x604;
    int2hexbytes(canmaster->cmsg_tx[0], w4_int);
    canmaster->transmit();
    return 0;
}