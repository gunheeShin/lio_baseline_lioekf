# MIT License
#
# Copyright (c) 2023 Yibin Wu, Tiziano Guadagnino
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
cmake_minimum_required(VERSION 3.16...3.24)
# cmake_minimum_required(VERSION 2.8.3)
project(lio_ekf)

SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
find_package(Eigen3 REQUIRED)

find_package(
  catkin REQUIRED
  COMPONENTS roscpp
             geometry_msgs
             nav_msgs
             sensor_msgs
             rosbag
             std_msgs
             tf2
             tf2_ros
             pcl_ros)
find_package(PCL 1.8 REQUIRED)
catkin_package(DEPENDS PCL)

include(kiss-icp.cmake)
# ROS1 node
add_executable(lio_ekf_node ${CMAKE_CURRENT_SOURCE_DIR}/src/lio_ekf.cpp 
                            ${CMAKE_CURRENT_SOURCE_DIR}/src/lio_ekf_node.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/src/OdometryServer.cpp 
                            ${CMAKE_CURRENT_SOURCE_DIR}/src/imuPropagation.cpp)
target_include_directories(lio_ekf_node PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${catkin_INCLUDE_DIRS})
 
target_link_libraries(lio_ekf_node 
                      kiss_icp::core 
                      Eigen3::Eigen 
                      ${catkin_LIBRARIES} 
                      ${PCL_LIBRARIES} )

install(TARGETS lio_ekf_node RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

target_link_libraries(lio_ekf_node /workspace/lio_baselines_ws/src/lio_baselines/include/data_recorder/build/libdata_recorder.so)
target_include_directories(lio_ekf_node PRIVATE /workspace/lio_baselines_ws/src/lio_baselines/include/data_recorder)
