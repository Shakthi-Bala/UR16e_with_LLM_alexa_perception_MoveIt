cmake_minimum_required(VERSION 3.8)
project(ur_package_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Required dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(action_msgs REQUIRED)

# Declare interface files
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ObjectPosition.msg"
  "msg/ObjectPositions.msg"
  "srv/GetObjectLocations.srv"
  "srv/GetSpeech.srv"           
  "action/PickAction.action"
  DEPENDENCIES geometry_msgs sensor_msgs action_msgs
)

# Install interface files
install(DIRECTORY msg srv action
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)

# Testing (optional)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
