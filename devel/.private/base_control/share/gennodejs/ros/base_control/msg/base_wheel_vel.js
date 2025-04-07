// Auto-generated. Do not edit!

// (in-package base_control.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class base_wheel_vel {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.w1 = null;
      this.w2 = null;
      this.w3 = null;
      this.w4 = null;
    }
    else {
      if (initObj.hasOwnProperty('w1')) {
        this.w1 = initObj.w1
      }
      else {
        this.w1 = 0.0;
      }
      if (initObj.hasOwnProperty('w2')) {
        this.w2 = initObj.w2
      }
      else {
        this.w2 = 0.0;
      }
      if (initObj.hasOwnProperty('w3')) {
        this.w3 = initObj.w3
      }
      else {
        this.w3 = 0.0;
      }
      if (initObj.hasOwnProperty('w4')) {
        this.w4 = initObj.w4
      }
      else {
        this.w4 = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type base_wheel_vel
    // Serialize message field [w1]
    bufferOffset = _serializer.float32(obj.w1, buffer, bufferOffset);
    // Serialize message field [w2]
    bufferOffset = _serializer.float32(obj.w2, buffer, bufferOffset);
    // Serialize message field [w3]
    bufferOffset = _serializer.float32(obj.w3, buffer, bufferOffset);
    // Serialize message field [w4]
    bufferOffset = _serializer.float32(obj.w4, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type base_wheel_vel
    let len;
    let data = new base_wheel_vel(null);
    // Deserialize message field [w1]
    data.w1 = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [w2]
    data.w2 = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [w3]
    data.w3 = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [w4]
    data.w4 = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 16;
  }

  static datatype() {
    // Returns string type for a message object
    return 'base_control/base_wheel_vel';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '17707310464ac4213ef0607f2a232ab3';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float32 w1
    float32 w2
    float32 w3
    float32 w4
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new base_wheel_vel(null);
    if (msg.w1 !== undefined) {
      resolved.w1 = msg.w1;
    }
    else {
      resolved.w1 = 0.0
    }

    if (msg.w2 !== undefined) {
      resolved.w2 = msg.w2;
    }
    else {
      resolved.w2 = 0.0
    }

    if (msg.w3 !== undefined) {
      resolved.w3 = msg.w3;
    }
    else {
      resolved.w3 = 0.0
    }

    if (msg.w4 !== undefined) {
      resolved.w4 = msg.w4;
    }
    else {
      resolved.w4 = 0.0
    }

    return resolved;
    }
};

module.exports = base_wheel_vel;
