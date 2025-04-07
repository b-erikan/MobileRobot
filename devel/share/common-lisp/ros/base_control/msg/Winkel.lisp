; Auto-generated. Do not edit!


(cl:in-package base_control-msg)


;//! \htmlinclude Winkel.msg.html

(cl:defclass <Winkel> (roslisp-msg-protocol:ros-message)
  ((rotate
    :reader rotate
    :initarg :rotate
    :type cl:float
    :initform 0.0))
)

(cl:defclass Winkel (<Winkel>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Winkel>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Winkel)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name base_control-msg:<Winkel> is deprecated: use base_control-msg:Winkel instead.")))

(cl:ensure-generic-function 'rotate-val :lambda-list '(m))
(cl:defmethod rotate-val ((m <Winkel>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader base_control-msg:rotate-val is deprecated.  Use base_control-msg:rotate instead.")
  (rotate m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Winkel>) ostream)
  "Serializes a message object of type '<Winkel>"
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'rotate))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Winkel>) istream)
  "Deserializes a message object of type '<Winkel>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'rotate) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Winkel>)))
  "Returns string type for a message object of type '<Winkel>"
  "base_control/Winkel")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Winkel)))
  "Returns string type for a message object of type 'Winkel"
  "base_control/Winkel")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Winkel>)))
  "Returns md5sum for a message object of type '<Winkel>"
  "be12cf1d559af9e1d7c33b2780105139")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Winkel)))
  "Returns md5sum for a message object of type 'Winkel"
  "be12cf1d559af9e1d7c33b2780105139")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Winkel>)))
  "Returns full string definition for message of type '<Winkel>"
  (cl:format cl:nil "float32 rotate~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Winkel)))
  "Returns full string definition for message of type 'Winkel"
  (cl:format cl:nil "float32 rotate~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Winkel>))
  (cl:+ 0
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Winkel>))
  "Converts a ROS message object to a list"
  (cl:list 'Winkel
    (cl:cons ':rotate (rotate msg))
))
