/**
This file decleares an inteface to CAN port, to faciliate farme transmission and reception
template: https://github.com/matthiasbock/libcan/
**/

#if(!defined(SOCKETNTCAN_H))
#define SOCKETNTCAN_H

#include "ntcan.h"

#include <pthread.h>

/*
Frame transmission and reception via CAN port
*/
class SocketNTCAN
{
    private:
    int net;                    // net number (can0)
    uint32_t mode;              // mode bits for canOpen()
    int32_t txqueuesize;        // maximum number of messages to transmit
    int32_t rxqueuesize;        // maximum number of messages to receive
    int32_t txtimeout;    	// timeout for transmit in ms
    int32_t rxtimeout;  	// timeout for receiving data in ms
    NTCAN_HANDLE handle;        // CAN handle returned by canOpen()
    NTCAN_RESULT retvalue;      // return values of NTCAN Api Calls
    uint32_t baud;              // CAN baudrate, 0=1000k Bits
    
    pthread_t receiver_thread_id;   
    
    public:
    int rtr;
    int canID;  // CAN ID for canRead() filter
    bool terminate_receiver_thread;
    CMSG cmsg_tx[8];            // buffer for can messages
    CMSG cmsg_rx[8];            // buffer for can messages

    // constructor
    SocketNTCAN();
    // destructor
    ~SocketNTCAN();

    // open and bind socket
    int open();
    
    // close and unbind socket
    int close();

    // send data
    int transmit_012(); // send default data 0 1 2
    int transmit(); // send default data 0 1 2
    // receive data thread
    int start_receiver_thread(int canID);
};

#endif
