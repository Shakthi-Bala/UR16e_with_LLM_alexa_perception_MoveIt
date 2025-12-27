cmake_minimum_required(VERSION 3.8)
project(alexa_vs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Required dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(ur_package_msgs REQUIRED) 
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# (No rosidl_generate_interfaces here!)

ament_export_dependencies(rosidl_default_runtime ur_package_msgs)

# Testing (optional)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
