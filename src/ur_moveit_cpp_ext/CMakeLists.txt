cmake_minimum_required(VERSION 3.8)
project(ur_moveit_cpp_ext)

# Enable compiler warnings
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(geometry_msgs REQUIRED)

# Declare executable
add_executable(moveit_xyz src/moveit_xyz.cpp)

# Link dependencies to executable
ament_target_dependencies(moveit_xyz
  rclcpp
  moveit_ros_planning_interface
  geometry_msgs
)

# Install executable
install(TARGETS
  moveit_xyz
  DESTINATION lib/${PROJECT_NAME}
)

# Enable tests and linting
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# Package manifest
ament_package()
