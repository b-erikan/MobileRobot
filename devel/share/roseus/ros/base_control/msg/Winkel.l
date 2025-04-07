;; Auto-generated. Do not edit!


(when (boundp 'base_control::Winkel)
  (if (not (find-package "BASE_CONTROL"))
    (make-package "BASE_CONTROL"))
  (shadow 'Winkel (find-package "BASE_CONTROL")))
(unless (find-package "BASE_CONTROL::WINKEL")
  (make-package "BASE_CONTROL::WINKEL"))

(in-package "ROS")
;;//! \htmlinclude Winkel.msg.html


(defclass base_control::Winkel
  :super ros::object
  :slots (_rotate ))

(defmethod base_control::Winkel
  (:init
   (&key
    ((:rotate __rotate) 0.0)
    )
   (send-super :init)
   (setq _rotate (float __rotate))
   self)
  (:rotate
   (&optional __rotate)
   (if __rotate (setq _rotate __rotate)) _rotate)
  (:serialization-length
   ()
   (+
    ;; float32 _rotate
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _rotate
       (sys::poke _rotate (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _rotate
     (setq _rotate (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get base_control::Winkel :md5sum-) "be12cf1d559af9e1d7c33b2780105139")
(setf (get base_control::Winkel :datatype-) "base_control/Winkel")
(setf (get base_control::Winkel :definition-)
      "float32 rotate
")



(provide :base_control/Winkel "be12cf1d559af9e1d7c33b2780105139")


