; Auto-generated. Do not edit!


(cl:in-package base_control-msg)


;//! \htmlinclude base_wheel_vel.msg.html

(cl:defclass <base_wheel_vel> (roslisp-msg-protocol:ros-message)
  ((w1
    :reader w1
    :initarg :w1
    :type cl:float
    :initform 0.0)
   (w2
    :reader w2
    :initarg :w2
    :type cl:float
    :initform 0.0)
   (w3
    :reader w3
    :initarg :w3
    :type cl:float
    :initform 0.0)
   (w4
    :reader w4
    :initarg :w4
    :type cl:float
    :initform 0.0))
)

(cl:defclass base_wheel_vel (<base_wheel_vel>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <base_wheel_vel>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'base_wheel_vel)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name base_control-msg:<base_wheel_vel> is deprecated: use base_control-msg:base_wheel_vel instead.")))

(cl:ensure-generic-function 'w1-val :lambda-list '(m))
(cl:defmethod w1-val ((m <base_wheel_vel>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader base_control-msg:w1-val is deprecated.  Use base_control-msg:w1 instead.")
  (w1 m))

(cl:ensure-generic-function 'w2-val :lambda-list '(m))
(cl:defmethod w2-val ((m <base_wheel_vel>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader base_control-msg:w2-val is deprecated.  Use base_control-msg:w2 instead.")
  (w2 m))

(cl:ensure-generic-function 'w3-val :lambda-list '(m))
(cl:defmethod w3-val ((m <base_wheel_vel>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader base_control-msg:w3-val is deprecated.  Use base_control-msg:w3 instead.")
  (w3 m))

(cl:ensure-generic-function 'w4-val :lambda-list '(m))
(cl:defmethod w4-val ((m <base_wheel_vel>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader base_control-msg:w4-val is deprecated.  Use base_control-msg:w4 instead.")
  (w4 m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <base_wheel_vel>) ostream)
  "Serializes a message object of type '<base_wheel_vel>"
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'w1))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'w2))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'w3))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'w4))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <base_wheel_vel>) istream)
  "Deserializes a message object of type '<base_wheel_vel>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'w1) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'w2) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'w3) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'w4) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<base_wheel_vel>)))
  "Returns string type for a message object of type '<base_wheel_vel>"
  "base_control/base_wheel_vel")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'base_wheel_vel)))
  "Returns string type for a message object of type 'base_wheel_vel"
  "base_control/base_wheel_vel")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<base_wheel_vel>)))
  "Returns md5sum for a message object of type '<base_wheel_vel>"
  "17707310464ac4213ef0607f2a232ab3")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'base_wheel_vel)))
  "Returns md5sum for a message object of type 'base_wheel_vel"
  "17707310464ac4213ef0607f2a232ab3")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<base_wheel_vel>)))
  "Returns full string definition for message of type '<base_wheel_vel>"
  (cl:format cl:nil "float32 w1~%float32 w2~%float32 w3~%float32 w4~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'base_wheel_vel)))
  "Returns full string definition for message of type 'base_wheel_vel"
  (cl:format cl:nil "float32 w1~%float32 w2~%float32 w3~%float32 w4~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <base_wheel_vel>))
  (cl:+ 0
     4
     4
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <base_wheel_vel>))
  "Converts a ROS message object to a list"
  (cl:list 'base_wheel_vel
    (cl:cons ':w1 (w1 msg))
    (cl:cons ':w2 (w2 msg))
    (cl:cons ':w3 (w3 msg))
    (cl:cons ':w4 (w4 msg))
))
