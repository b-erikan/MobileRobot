cmake_minimum_required(VERSION 3.0.2)
project(motion_control)

set(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/src/eigen")
#set(CASADI_DIR "${CMAKE_SOURCE_DIR}/src/casadi")
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(casadi REQUIRED)

# Find catkin macros and dependencies
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  message_generation
  base_control
)

# Add your message files
#add_message_files(
#  FILES
#  base_wheel_vel.msg
#)

# Generate the messages
generate_messages(
  DEPENDENCIES
  std_msgs  # You can add more dependencies here if necessary
  #base_wheel_vel.msg
)

# Declare catkin package and dependencies
catkin_package(
  CATKIN_DEPENDS roscpp std_msgs message_runtime base_control
)

# Include directories
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  #${CASADI_INCLUDE_DIRS}
)

# Add your executables here, e.g., a ROS node
add_executable(motion_control_start src/motion_control.cpp)

# Link the executable with the required libraries
target_link_libraries(motion_control_start
  ${catkin_LIBRARIES}
  #${CASADI_LIBRARIES}
  casadi
)

# Add message headers as dependencies for your node
add_dependencies(motion_control_start ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
