;; Auto-generated. Do not edit!


(when (boundp 'base_control::base_wheel_vel)
  (if (not (find-package "BASE_CONTROL"))
    (make-package "BASE_CONTROL"))
  (shadow 'base_wheel_vel (find-package "BASE_CONTROL")))
(unless (find-package "BASE_CONTROL::BASE_WHEEL_VEL")
  (make-package "BASE_CONTROL::BASE_WHEEL_VEL"))

(in-package "ROS")
;;//! \htmlinclude base_wheel_vel.msg.html


(defclass base_control::base_wheel_vel
  :super ros::object
  :slots (_w1 _w2 _w3 _w4 ))

(defmethod base_control::base_wheel_vel
  (:init
   (&key
    ((:w1 __w1) 0.0)
    ((:w2 __w2) 0.0)
    ((:w3 __w3) 0.0)
    ((:w4 __w4) 0.0)
    )
   (send-super :init)
   (setq _w1 (float __w1))
   (setq _w2 (float __w2))
   (setq _w3 (float __w3))
   (setq _w4 (float __w4))
   self)
  (:w1
   (&optional __w1)
   (if __w1 (setq _w1 __w1)) _w1)
  (:w2
   (&optional __w2)
   (if __w2 (setq _w2 __w2)) _w2)
  (:w3
   (&optional __w3)
   (if __w3 (setq _w3 __w3)) _w3)
  (:w4
   (&optional __w4)
   (if __w4 (setq _w4 __w4)) _w4)
  (:serialization-length
   ()
   (+
    ;; float32 _w1
    4
    ;; float32 _w2
    4
    ;; float32 _w3
    4
    ;; float32 _w4
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _w1
       (sys::poke _w1 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _w2
       (sys::poke _w2 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _w3
       (sys::poke _w3 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _w4
       (sys::poke _w4 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _w1
     (setq _w1 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _w2
     (setq _w2 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _w3
     (setq _w3 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _w4
     (setq _w4 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get base_control::base_wheel_vel :md5sum-) "17707310464ac4213ef0607f2a232ab3")
(setf (get base_control::base_wheel_vel :datatype-) "base_control/base_wheel_vel")
(setf (get base_control::base_wheel_vel :definition-)
      "float32 w1
float32 w2
float32 w3
float32 w4

")



(provide :base_control/base_wheel_vel "17707310464ac4213ef0607f2a232ab3")


