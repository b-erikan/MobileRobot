
(cl:in-package :asdf)

(defsystem "base_control-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "Winkel" :depends-on ("_package_Winkel"))
    (:file "_package_Winkel" :depends-on ("_package"))
    (:file "base_wheel_vel" :depends-on ("_package_base_wheel_vel"))
    (:file "_package_base_wheel_vel" :depends-on ("_package"))
  ))