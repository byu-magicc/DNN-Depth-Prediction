cmake_minimum_required(VERSION 2.8.3)
project(ttc_object_avoidance)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_FLAGS "-fopenmp")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  geometry_msgs
  nav_msgs
  cv_bridge
  image_transport
  message_generation
)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 4.4.0 REQUIRED)
find_package(Ceres REQUIRED)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

# Generate dynamic reconfigure parameters in the 'cfg' folder
#generate_dynamic_reconfigure_options(
#
#)

add_message_files(
  DIRECTORY msg
  FILES
  FeatData.msg
  TrackedFeats.msg
  FeatAndDisplacement.msg
  TrackedFeatsWDis.msg
  FeatAndDepth.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp sensor_msgs geometry_msgs nav_msgs cv_bridge image_transport message_runtime
)

include_directories(include 
  ${catkin_INCLUDE_DIRS} 
  ${EIGEN3_INCLUDE_DIRS} 
  ${CERES_INCLUDE_DIRS})

add_executable(feature_tracker
            src/vision/feature_tracker_node.cpp
            src/vision/feature_tracker.cpp
            src/vision/gyro_integrator.cpp)
add_executable(depth_predictor
            src/depth/depth_predictor_node.cpp
            src/depth/depth_predictor.cpp)
add_dependencies(feature_tracker ${catkin_EXPORTED_TARGETS})
add_dependencies(depth_predictor ${catkin_EXPORTED_TARGETS})
target_link_libraries(feature_tracker ${catkin_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(depth_predictor ${catkin_LIBRARIES})