#pragma once

#include <vector>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <cmath>
#include <chrono>

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "lie_groups/so3.hpp"
#include "lie_groups/se3.hpp"
#include "vision/feature_manager.h"
#include "vision/gyro_integrator.h"

namespace vision
{

class FeatureTracker
{
public:
  FeatureTracker();

  void set_intrinsic(cv::Mat K, cv::Mat D, int img_w, int img_h);
  void set_rotation_compensation_params(lie_groups::SO3d R_bc, std::shared_ptr<GyroIntegrator> gyro_integrator);
  void set_gf2t_params(int max_corners, double quality_level, int min_distance, int subpix_win_size, int subpix_max_iter, double subpix_eps);
  void set_klt_params(int win_size, int max_level, int max_iter, int eps);
  void setup_keyframes(int window_length, double parallax_threshold, int minimum_features, int minimum_keyframes_for_inclusion);
  void setup_image_partitions(int part_w, int part_h);
  void clear_all_features();
  void get_draw_image(cv::Mat& draw_image);

  std::shared_ptr<std::map<int, Feature>> process(cv::Mat const& new_image, int image_id, double image_time, bool& new_keyframe);

private:
  void update_relative_rotation();
  void process_current_image();
  void find_new_features(std::vector<cv::Point2f>& feat_new);
  void find_optic_flow();
  void filter_matches(std::vector<cv::Point2f> const& p_i, std::vector<cv::Point2f> const& p_j, std::vector<uchar>& status);
  void filter_points(std::vector<cv::Point2f> const& pts, std::vector<uchar> const& status, std::vector<cv::Point2f>& filtered_pts);
  void update_keyframes();
  void create_mask(cv::Mat& mask);
  void find_visible_pts_per_part(std::vector<int>& visible_pts_per_part);

  // image parameters
  cv::Mat current_img_, prev_img_;
  int current_img_id_;
  double current_img_time_;
  int img_w_;
  int img_h_;
  int num_part_w_;
  int num_part_h_;

  // parallax rotation compensation parameters
  bool use_rotation_compensated_parallax_;
  std::shared_ptr<GyroIntegrator> gyro_integrator_; // retrieves relative rotation from gyro measurements
  lie_groups::SO3d R_bc_; //rotation from body to camera

  // camera parameters
  cv::Mat K_;
  cv::Mat K_inv_;
  cv::Mat D_;
  std::vector<cv::Rect> part_rects_;

  // feature manager
  std::unique_ptr<FeatureManager> feat_manager_;

  // good features to track parameters
  int gf2t_max_corners_;
  double gf2t_quality_level_;
  int gf2t_min_distance_;
  cv::Size subpix_win_size_;
  int subpix_max_iter_;
  double subpix_eps_;
  cv::TermCriteria subpix_criteria_;

  // klt parameters
  cv::Size klt_win_size_;
  int klt_max_level_;
  int klt_max_iter_;
  double klt_eps_;
  cv::TermCriteria klt_criteria_;

  // keyframe parameters
  int keyframe_window_length_;
  double keyframe_parallax_threshold_;
  int keyframe_minimum_features_;
  int minimum_keyframes_for_inclusion_;
  bool new_keyframe_declared_;
  lie_groups::SO3d R_rel_; // approximate relative rotation since last keyframe

  std::vector<cv::Point2f> visible_pts_;

};

} //namespace vision
