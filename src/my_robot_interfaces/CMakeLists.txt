cmake_minimum_required(VERSION 3.8)
project(my_robot_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
# 1. ADD THIS LINE:
find_package(geometry_msgs REQUIRED) 

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/RunVision.action"
  # 2. ADD THIS DEPENDENCIES SECTION:
  DEPENDENCIES geometry_msgs
)

ament_package()