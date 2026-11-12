cmake_minimum_required(VERSION 3.8)
project(offroad_nav2)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TrafficLightState.msg"
  DEPENDENCIES std_msgs)

# Python packages (pure numpy/torch cores + ROS nodes)
ament_python_install_package(offroad_perception)
ament_python_install_package(offroad_vehicle_interface)
ament_python_install_package(offroad_bringup)

# Executables (`ros2 run offroad_nav2 <name>`) and the test/diagnostic tools
file(GLOB EXECUTABLES ${CMAKE_CURRENT_SOURCE_DIR}/scripts/*)
install(PROGRAMS ${EXECUTABLES} DESTINATION lib/${PROJECT_NAME})
file(GLOB TOOLS ${CMAKE_CURRENT_SOURCE_DIR}/tools/*.py)
install(PROGRAMS ${TOOLS} DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config behavior_trees urdf DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(offroad_nav2_unit test
    PYTHON_EXECUTABLE python3
    TIMEOUT 300)
endif()

ament_package()
