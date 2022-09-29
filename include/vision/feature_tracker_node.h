#pragma once

#include <vector>
#include <memory>
#include <string>
#include <map>

#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/Point.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include "spline_vio/FeatData.h"
#include "spline_vio/TrackedFeats.h"
#include <Eigen/Dense>

#include "lie_groups/so3.hpp"
#include "vision/feature_tracker.h"
#include "vision/gyro_integrator.h"

namespace vision
{

class FeatureTrackerNode
{
public:
  FeatureTrackerNode();
  float spin();

private:
	void image_callback(const sensor_msgs::ImageConstPtr img_msg);
  void imu_callback(const sensor_msgs::ImuConstPtr msg);

  // TODO: void bias_callback(const msg);

  void setup_camera();
  void setup_gf2t();
  void setup_klt();
  void setup_image_partitions();
  void setup_keyframes();

	void pub_all_features(std::shared_ptr<std::map<int, Feature>> feats, const std_msgs::Header header);
  void pub_keyframe_features(std::shared_ptr<std::map<int, Feature>> feats, const std_msgs::Header header, bool new_keyframe);

  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;
  image_transport::ImageTransport it_;
  ros::Subscriber image_sub_;
  ros::Subscriber imu_sub_;
  // ros::Subscriber bias_sub_;
  image_transport::Publisher draw_img_pub_;
  ros::Publisher feat_pub_;

  bool pub_draw_feats_;
  bool pub_at_camera_rate_;
  bool use_keyframes_;

  std::unique_ptr<FeatureTracker> feature_tracker_;
  std::shared_ptr<GyroIntegrator> gyro_integrator_;
  double use_rotation_compensated_parallax_;

  bool initialized_;
  double start_time_;
};

} // namespace vision