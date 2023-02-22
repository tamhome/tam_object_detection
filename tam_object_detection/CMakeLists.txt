cmake_minimum_required(VERSION 3.0.2)
project(tam_object_detection)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  message_generation
)

add_message_files(
  FILES
  BBox.msg
  Pose3D.msg
  ObjectDetection.msg
)

add_service_files(
  FILES
  ObjectDetectionService.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)