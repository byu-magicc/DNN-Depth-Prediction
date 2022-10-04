#include "vision/feature_tracker_node.h"

namespace vision
{

FeatureTrackerNode::FeatureTrackerNode() :
  nh_{}, nh_private_{"~"}, it_{nh_}, feature_tracker_{new FeatureTracker()}, use_rotation_compensated_parallax_{false}
{
  std::string img_topic, imu_topic;
  if(!nh_private_.getParam("/img_topic", img_topic) || !nh_private_.getParam("/imu_topic", imu_topic))
    ROS_FATAL("Need image topic name!!");

  image_sub_ = nh_.subscribe(img_topic, 1, &FeatureTrackerNode::image_callback, this);

  feat_pub_ = nh_.advertise<ttc_object_avoidance::TrackedFeats>("tracked_features", 10);

  nh_private_.param<bool>("/pub_draw_feats", pub_draw_feats_, true);
  if(pub_draw_feats_)
    draw_img_pub_ = it_.advertise("features_image", 1);

  setup_camera();
  setup_gf2t();
  setup_klt();
  setup_image_partitions();
  setup_keyframes();
}

void FeatureTrackerNode::setup_camera()
{
  // get camera params
  double k1, k2, p1, p2, fx, fy, cx, cy; 
  nh_private_.param<double>("/distortion_parameters/k1", k1, 0.0);
  nh_private_.param<double>("/distortion_parameters/k2", k2, 0.0);
  nh_private_.param<double>("/distortion_parameters/p1", p1, 0.0);
  nh_private_.param<double>("/distortion_parameters/p2", p2, 0.0);
  nh_private_.param<double>("/projection_parameters/fx", fx, 600.0);
  nh_private_.param<double>("/projection_parameters/fy", fy, 600.0);
  nh_private_.param<double>("/projection_parameters/cx", cx, 320.0);
  nh_private_.param<double>("/projection_parameters/cy", cy, 240.0);
  cv::Mat K = (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
  cv::Mat D = (cv::Mat_<double>(4,1) << k1, k2, p1, p2);

  int img_w, img_h;
  nh_private_.param<int>("/img_w", img_w, 640);
  nh_private_.param<int>("/img_h", img_h, 480);

  feature_tracker_->set_intrinsic(K, D, img_w, img_h);

  // get imu params for rotation compensation
  bool use_rotation_compensated_parallax_ = false;
  nh_private_.param<bool>("/keyframe/use_rotation_compensated_parallax", use_rotation_compensated_parallax_, false);

  if(use_rotation_compensated_parallax_)
  {
    std::string imu_topic;//, bias_topic;
    if(!nh_private_.getParam("/imu_topic", imu_topic)) // || !nh_private_.getParam("/bias_topic", bias_topic))
      ROS_FATAL("If use_compensated_parallax is true, must provide imu topic name!");

    imu_sub_ = nh_.subscribe(imu_topic, 1, &FeatureTrackerNode::imu_callback, this);
    // bias_sub_ = nh_.subscribe(bias_topic, 1, &FeatureTrackerNode::bias_callback, this);

    std::vector<double> T_bc_params;
    nh_private_.param<std::vector<double>>("/imu_cam_extrinsics", T_bc_params, T_bc_params);
    Eigen::Matrix4d T_bc = Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(T_bc_params.data());
    lie_groups::SO3d R_bc = lie_groups::SO3d(T_bc.block<3,3>(0,0).eval());

    double t_delta_imu = 0.0;
    nh_private_.param<double>("/imu_cam_time_offset", t_delta_imu, t_delta_imu);
    gyro_integrator_.reset(new GyroIntegrator(t_delta_imu));
    
    feature_tracker_->set_rotation_compensation_params(R_bc, gyro_integrator_);
  }
}

void FeatureTrackerNode::setup_gf2t()
{
  int max_corners, min_distance, subpix_win_size, subpix_max_iter;
  double quality_level, subpix_eps;
  nh_private_.param<int>("/gf2t/max_corners", max_corners, 100);
  nh_private_.param<double>("/gf2t/quality_level", quality_level, 0.05);
  nh_private_.param<int>("/gf2t/min_distance", min_distance, 5);
  nh_private_.param<int>("/gf2t/subpix_win_size", subpix_win_size, 10);
  nh_private_.param<int>("/gf2t/subpix_max_iter", subpix_max_iter, 10);
  nh_private_.param<double>("/gf2t/subpix_eps", subpix_eps, 0.03);

  feature_tracker_->set_gf2t_params(max_corners, quality_level, min_distance, subpix_win_size, subpix_max_iter, subpix_eps);
}

void FeatureTrackerNode::setup_klt()
{
  int win_size, max_level, max_iter;
  double eps;
  nh_private_.param<int>("/klt/win_size", win_size, 31);
  nh_private_.param<int>("/klt/max_level", max_level, 3);
  nh_private_.param<int>("/klt/max_iter", max_iter, 30);
  nh_private_.param<double>("/klt/eps", eps, 0.03);

  feature_tracker_->set_klt_params(win_size, max_level, max_iter, eps);
}

void FeatureTrackerNode::setup_image_partitions()
{
  int part_w, part_h;
  nh_private_.param<int>("/part_w", part_w, 1);
  nh_private_.param<int>("/part_h", part_h, 1);

  feature_tracker_->setup_image_partitions(part_w, part_h);
}

void FeatureTrackerNode::setup_keyframes()
{
  int window_length, minimum_features, minimum_keyframes_for_inclusion;
  double parallax_threshold;
  nh_private_.param<int>("/keyframe/window_length", window_length, 10);
  nh_private_.param<double>("/keyframe/keyframe_parallax_threshold", parallax_threshold, 30.0);
  nh_private_.param<int>("/keyframe/minimum_features", minimum_features, 100);
  nh_private_.param<int>("/keyframe/minimum_keyframes_for_inclusion", minimum_keyframes_for_inclusion, 2);
  nh_private_.param<bool>("/keyframe/pub_at_camera_rate", pub_at_camera_rate_, false); // if true, features from the most recent camera
                                                                                       // frame will be appended to the keyframe features
                                                                                       // to allow camera frame-rate estimation
  nh_private_.param<bool>("/use_keyframes", use_keyframes_, true); // if false, camera features from every image in the sliding window will be
                                                                   // published, not just keyframes

  feature_tracker_->setup_keyframes(window_length, parallax_threshold, minimum_features, minimum_keyframes_for_inclusion);
}

void FeatureTrackerNode::image_callback(const sensor_msgs::ImageConstPtr img_msg)
{
  if(!initialized_)
  {
    start_time_ = img_msg->header.stamp.toSec();
    initialized_ = true;
  }

  cv_bridge::CvImageConstPtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
  }
  catch(cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  double img_time = img_msg->header.stamp.toSec() - start_time_;
  bool new_keyframe;
  std::shared_ptr<std::map<int, Feature>> feats = feature_tracker_->process(cv_ptr->image, img_msg->header.seq, img_time, new_keyframe);

  if(!use_keyframes_) pub_all_features(feats, img_msg->header);
  else
  {
    if(new_keyframe || pub_at_camera_rate_)
      pub_keyframe_features(feats, img_msg->header, new_keyframe);
  }

  if(pub_draw_feats_)
  {
    cv_bridge::CvImage img_bridge;
    sensor_msgs::Image draw_img_msg;
    cv::Mat draw_img;
    feature_tracker_->get_draw_image(draw_img);
    img_bridge = cv_bridge::CvImage(img_msg->header, sensor_msgs::image_encodings::RGB8, draw_img);
    img_bridge.toImageMsg(draw_img_msg);
    draw_img_pub_.publish(draw_img_msg);
  }
}

void FeatureTrackerNode::imu_callback(const sensor_msgs::ImuConstPtr msg)
{
  if(!initialized_) return;

  double msg_time = msg->header.stamp.toSec() - start_time_;
  Eigen::Vector3d gyro = (Eigen::Vector3d() << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z).finished();
  gyro_integrator_->insert_new_measurement(std::make_pair(msg_time, gyro));
}

// void FeatureTrackerNode::bias_callback(const msg)
// {
//   if(!initialized_) return;

//   Eigen::Vector3d gyro_bias = (Eigen::Vector3d() << msg->gyro_bias.x, msg->gyro_bias.y, msg->gyro_bias.z).finished();
//   gyro_integrator_->update_bias(gyro_bias);
// }

void FeatureTrackerNode::pub_all_features(std::shared_ptr<std::map<int, Feature>> feats, const std_msgs::Header header)
{
  ttc_object_avoidance::TrackedFeats feats_msg;
  Feature feat;
  ttc_object_avoidance::FeatData feat_data;
  geometry_msgs::Point pt, norm_pt;

  for(auto const& map_feat : *feats)
  {
    feat = map_feat.second;
    if(!feat.active) continue;
    
    feat_data.pts.clear();
    feat_data.norm_pts.clear();
    feat_data.img_ids.clear();

    feat_data.feat_id = feat.feat_id;
    for(int i = 0; i < feat.pts.size(); ++i)
    {
      pt.x = feat.pts[i].x;
      pt.y = feat.pts[i].y;
      pt.z = 1.0;

      norm_pt.x = feat.norm_pts[i].x;
      norm_pt.y = feat.norm_pts[i].y;
      norm_pt.z = 1.0;

      feat_data.pts.push_back(pt);
      feat_data.norm_pts.push_back(norm_pt);
      feat_data.img_ids.push_back(feat.image_ids[i]);
    }

    feats_msg.feats.push_back(feat_data);
  }
  
  feats_msg.new_keyframe = false;
  feats_msg.header = header;
  feat_pub_.publish(feats_msg);
}

void FeatureTrackerNode::pub_keyframe_features(std::shared_ptr<std::map<int, Feature>> feats, const std_msgs::Header header, bool new_keyframe)
{
  ttc_object_avoidance::TrackedFeats keyframe_feats_msg;
  Feature feat;
  ttc_object_avoidance::FeatData feat_data;
  geometry_msgs::Point pt, norm_pt;

  for(auto &map_feat : *feats)
  {
    feat = map_feat.second;
    if(!feat.active) continue;
    
    feat_data.pts.clear();
    feat_data.norm_pts.clear();
    feat_data.img_ids.clear();

    feat_data.feat_id = feat.feat_id;
    for(int i = 0; i < feat.keyframe_pts.size(); i++)
    {
      pt.x = feat.keyframe_pts[i].second.first.x;
      pt.y = feat.keyframe_pts[i].second.first.y;
      pt.z = 1.0;

      norm_pt.x = feat.keyframe_pts[i].second.second.x;
      norm_pt.y = feat.keyframe_pts[i].second.second.y;
      norm_pt.z = 1.0;

      feat_data.pts.push_back(pt);
      feat_data.norm_pts.push_back(norm_pt);
      feat_data.img_ids.push_back(feat.keyframe_pts[i].first);
    }

    // add current image features if publishing at camera frame rate
    if(!new_keyframe)
    {
      pt.x = feat.pts.back().x;
      pt.y = feat.pts.back().y;
      pt.z = 1.0;

      norm_pt.x = feat.norm_pts.back().x;
      norm_pt.y = feat.norm_pts.back().y;
      norm_pt.z = 1.0;

      feat_data.pts.push_back(pt);
      feat_data.norm_pts.push_back(norm_pt);
      feat_data.img_ids.push_back(feat.image_ids.back());
    }

    keyframe_feats_msg.feats.push_back(feat_data);
  }

  keyframe_feats_msg.new_keyframe = new_keyframe;
  keyframe_feats_msg.header = header;
  feat_pub_.publish(keyframe_feats_msg);
}

} // namespace vision


int main(int argc, char **argv)
{
  ros::init(argc, argv, "feature_tracker");
  vision::FeatureTrackerNode *feat = new vision::FeatureTrackerNode();

  ros::spin();

  return 0;
}