cmake_minimum_required(VERSION 3.8)
project(vesta_neural_controller)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_util REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)

# ONNX Runtime (C++ API). docker/Dockerfile installs it to /opt/onnxruntime;
# elsewhere set ONNXRUNTIME_ROOT. Without it the plugin builds but only runs the fallback.
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS $ENV{ONNXRUNTIME_ROOT}/include /opt/onnxruntime/include
  PATH_SUFFIXES onnxruntime onnxruntime/core/session)
find_library(ONNXRUNTIME_LIBRARY onnxruntime
  HINTS $ENV{ONNXRUNTIME_ROOT}/lib /opt/onnxruntime/lib)

set(dependencies
  geometry_msgs nav2_core nav2_costmap_2d nav2_util nav_msgs pluginlib rclcpp rclcpp_lifecycle
  std_msgs tf2 tf2_ros visualization_msgs)

add_library(${PROJECT_NAME} SHARED
  src/neural_controller.cpp
  src/observation.cpp
  src/onnx_policy.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(${PROJECT_NAME} ${dependencies})

if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
  message(STATUS "onnxruntime: ${ONNXRUNTIME_LIBRARY}")
  target_compile_definitions(${PROJECT_NAME} PRIVATE VESTA_HAVE_ONNXRUNTIME)
  target_include_directories(${PROJECT_NAME} PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
  target_link_libraries(${PROJECT_NAME} ${ONNXRUNTIME_LIBRARY})
  get_filename_component(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_LIBRARY} DIRECTORY)
  set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}")
else()
  message(WARNING "onnxruntime not found: the neural controller will only run its fallback")
endif()

pluginlib_export_plugin_description_file(nav2_core plugin.xml)

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include/)
install(DIRECTORY models DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_observation test/test_observation.cpp src/observation.cpp)
  target_include_directories(test_observation PRIVATE include)
  target_compile_definitions(test_observation PRIVATE
    FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/test/fixtures/obs_cases.txt")
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(${dependencies})
ament_package()
