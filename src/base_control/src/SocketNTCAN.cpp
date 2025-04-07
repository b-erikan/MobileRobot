

#include <iostream>
#include "SocketNTCAN.h"

SocketNTCAN::SocketNTCAN(){
    net = 0; // can0
    mode = 0; // mode of the port
    txqueuesize = 8; 
    rxqueuesize = 8;
    txtimeout = 100;
    rxtimeout = 1000; 
    baud = 0; // bitmask, 0 = 1000 kBit/s
    rtr = 0; // rtr frame: 0 = dominant in data frame
}

SocketNTCAN::~SocketNTCAN(){
    std::cout << "Closing CAN port..." << std::endl;
    retvalue = canClose(handle);
    if (retvalue !=NTCAN_SUCCESS)
        std::cout << "canClose() failed with error " << retvalue << std::endl;
    else
        std::cout << "canClose() returned ok!" << std::endl;
}

int SocketNTCAN::open(){
    std::cout << "------------------------------------------" << std::endl;
    // open CAN bus port
    retvalue = canOpen(net, mode, txqueuesize, rxqueuesize, txtimeout, rxtimeout, &handle);
    if (retvalue!=NTCAN_SUCCESS){
        std::cout << "canOpen() failed with error " << retvalue << std::endl;
        return -1;
    }
    std::cout << "function canOpen() returned Ok!" << std::endl;
    std::cout << "--------------------" << std::endl;
    // set baud rate
    retvalue = canSetBaudrate(handle, baud);
    if (retvalue != 0){
        std::cout << "canSetBaudrate() failed with error " << retvalue << std::endl;
        canClose(handle);
        return -1;
    }
    std::cout << "function canSetBaudrate() returned Ok!" << std::endl;
    std::cout << "--------------------" << std::endl;
    return 0; // 0: successful
}

int SocketNTCAN::close(){
        retvalue = canClose(handle);
    if (retvalue !=NTCAN_SUCCESS){
        std::cout << "canClose failed with error " << retvalue << std::endl;
        return -1;
    }
    else
        std::cout << "canClose() returned ok!" << std::endl;
    return 0;
}

int SocketNTCAN::transmit(){
    // initialize first message buffer 

    int len = 1;  // number of valid messages in cmsg buffer
    retvalue = canWrite(handle, &cmsg_tx[0], &len, NULL);

    if(retvalue != NTCAN_SUCCESS){
        std::cout << "canWrite() failed with error " << retvalue << std::endl;
        return -1;
    }
    else
        //std::cout << "canWrite() returned ok!" << std::endl;
    return 0;
    //
}

int SocketNTCAN::transmit_012(){
    // initialize first message buffer 
    // CAN id = 0, len = 3, and data0 - data2 = 0,1,2
    cmsg_tx[0].id = 0x00;
    cmsg_tx[0].len = 0x03; 
    cmsg_tx[0].len |= cmsg_tx[0].len + (rtr<<4);
    for (int i = 0; i<3; i++)
        cmsg_tx[0].data[i] = i;
    int len = 1;  // number of valid messages in cmsg buffer
    retvalue = canWrite(handle, &cmsg_tx[0], &len, NULL);

    if(retvalue != NTCAN_SUCCESS){
        std::cout << "canWrite() failed with error " << retvalue << std::endl;
        return -1;
    }
    else
        std::cout << "canWrite() returned ok!" << std::endl;
    return 0;
    //
}