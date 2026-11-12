cmake_minimum_required(VERSION 3.8)
project(offroad_nav2)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wno-unused-parameter)
endif()

# ONLY_NODES="a;b" builds only those executables (plus the core library): used to compile a part of the package quickly.
set(ONLY_NODES "" CACHE STRING "build only these executables (semicolon separated); empty = all")

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(stereo_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TrafficLightState.msg"
  DEPENDENCIES std_msgs)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

# ONNX Runtime (GPU build unpacked by docker/Dockerfile into /opt/onnxruntime)
set(ORT_ROOT "$ENV{ORT_ROOT}" CACHE PATH "ONNX Runtime root (include/, lib/)")
if(NOT ORT_ROOT)
  set(ORT_ROOT /opt/onnxruntime)
endif()

# ---- core library: pure C++ (no ROS) geometry, post-processing, tracking, vehicle logic; the ONNX engine ----------------------
file(GLOB CORE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp)
list(FILTER CORE_SRCS EXCLUDE REGEX ".*/engine\\.cpp$")
add_library(offroad_core SHARED ${CORE_SRCS})
target_include_directories(offroad_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(offroad_core PUBLIC Eigen3::Eigen ${OpenCV_LIBS})

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/core/engine.cpp AND EXISTS ${ORT_ROOT}/include)
  add_library(offroad_engine SHARED src/core/engine.cpp)
  target_include_directories(offroad_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${ORT_ROOT}/include)
  target_link_libraries(offroad_engine PUBLIC offroad_core ${ORT_ROOT}/lib/libonnxruntime.so)
  set_target_properties(offroad_engine PROPERTIES INSTALL_RPATH "${ORT_ROOT}/lib")
  install(TARGETS offroad_engine LIBRARY DESTINATION lib)
  set(HAVE_ENGINE TRUE)
endif()
install(TARGETS offroad_core LIBRARY DESTINATION lib)

# ---- nodes: one executable per src/nodes/<name>.cpp (installed as `ros2 run offroad_nav2 <name>`) --------------------------------
set(ROS_DEPS rclcpp std_msgs geometry_msgs nav_msgs sensor_msgs stereo_msgs vision_msgs ackermann_msgs tf2 tf2_ros tf2_eigen)
file(GLOB NODE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/*.cpp)
set(CPP_NODE_NAMES "")
foreach(src ${NODE_SRCS})
  get_filename_component(name ${src} NAME_WE)
  if(ONLY_NODES AND NOT name IN_LIST ONLY_NODES)
    continue()
  endif()
  # nodes that need the inference engine are skipped when ONNX Runtime is not available
  file(READ ${src} _content)
  if(_content MATCHES "offroad_nav2/engine.hpp" AND NOT HAVE_ENGINE)
    message(WARNING "skipping ${name}: ONNX Runtime not found at ${ORT_ROOT}")
    continue()
  endif()
  add_executable(${name} ${src})
  target_link_libraries(${name} offroad_core ${cpp_typesupport_target})
  if(HAVE_ENGINE)
    target_link_libraries(${name} offroad_engine)
  endif()
  ament_target_dependencies(${name} ${ROS_DEPS})
  install(TARGETS ${name} DESTINATION lib/${PROJECT_NAME})
  list(APPEND CPP_NODE_NAMES ${name})
endforeach()

# ---- Python parts that remain: bringup tools (nav_goal, gps_route, lap_*), launch helpers, nav2_params.py ----------------------
ament_python_install_package(offroad_bringup)
file(GLOB SCRIPTS ${CMAKE_CURRENT_SOURCE_DIR}/scripts/*)
install(PROGRAMS ${SCRIPTS} DESTINATION lib/${PROJECT_NAME})
file(GLOB TOOLS ${CMAKE_CURRENT_SOURCE_DIR}/tools/*.py)
install(PROGRAMS ${TOOLS} DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config behavior_trees urdf DESTINATION share/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  file(GLOB TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp)
  foreach(t ${TEST_SRCS})
    get_filename_component(tname ${t} NAME_WE)
    if(ONLY_NODES AND NOT tname IN_LIST ONLY_NODES)
      continue()
    endif()
    ament_add_gtest(${tname} ${t})
    target_link_libraries(${tname} offroad_core)
    if(HAVE_ENGINE)
      target_link_libraries(${tname} offroad_engine)
    endif()
  endforeach()
  ament_add_pytest_test(offroad_nav2_python test
    PYTHON_EXECUTABLE python3
    TIMEOUT 300)
endif()

ament_package()
