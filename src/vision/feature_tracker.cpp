#include "vision/feature_tracker.h"

namespace vision
{

FeatureTracker::FeatureTracker():
  new_keyframe_declared_{false}, use_rotation_compensated_parallax_{false}, R_rel_{lie_groups::SO3d()}
{}

void FeatureTracker::set_intrinsic(cv::Mat K, cv::Mat D, int img_w, int img_h)
{
  K_ = K;
  K_inv_ = K.inv();
  D_ = D;

  img_w_ = img_w;
  img_h_ = img_h;
}

void FeatureTracker::set_rotation_compensation_params(lie_groups::SO3d R_bc, std::shared_ptr<GyroIntegrator> gyro_integrator)
{
  R_bc_ = R_bc;
  gyro_integrator_ = gyro_integrator;
  use_rotation_compensated_parallax_ = true;
}

void FeatureTracker::set_gf2t_params(int max_corners, double quality_level, int min_distance, int subpix_win_size, int subpix_max_iter, double subpix_eps)
{
  gf2t_max_corners_ = max_corners;
  gf2t_quality_level_ = quality_level;
  gf2t_min_distance_ = min_distance;
  subpix_max_iter_ = subpix_max_iter;
  subpix_eps_ = subpix_eps;

  subpix_win_size_ = cv::Size(subpix_win_size, subpix_win_size);
  subpix_criteria_ = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, subpix_max_iter_, subpix_eps_);
}

void FeatureTracker::set_klt_params(int win_size, int max_level, int max_iter, int eps)
{
  klt_max_level_ = max_level;
  klt_max_iter_ = max_iter;
  klt_eps_ = eps;

  klt_win_size_ = cv::Size(win_size, win_size);
  klt_criteria_ = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, klt_max_iter_, klt_eps_);
}

void FeatureTracker::setup_image_partitions(int part_w, int part_h)
{
  num_part_w_ = part_w;
  num_part_h_ = part_h;

  int box_w = img_w_/part_w;
  int box_h = img_h_/part_h;

  int x, y, w, h;

  for(int i = 0; i < part_h; i++)
  {
    for(int j = 0; j < part_w; j++)
    {
      x = j * box_w;
      if(j == (part_w - 1))
        w = img_w_ - x;
      else
        w = box_w;
      y = i * box_h;
      if(i == (part_h - 1))
        h = img_h_ - y;
      else
        h = box_h;
      cv::Rect new_part(x,y,w,h);
      part_rects_.push_back(new_part);
    }
  }
}

void FeatureTracker::setup_keyframes(int window_length, double parallax_threshold, int minimum_features, int minimum_keyframes_for_inclusion)
{
  keyframe_window_length_ = window_length;
  keyframe_parallax_threshold_ = parallax_threshold;
  keyframe_minimum_features_ = minimum_features;
  minimum_keyframes_for_inclusion_ = minimum_keyframes_for_inclusion;

  feat_manager_.reset(new FeatureManager(K_, keyframe_window_length_, keyframe_parallax_threshold_, keyframe_minimum_features_, minimum_keyframes_for_inclusion_));
}

std::shared_ptr<std::map<int, Feature>> FeatureTracker::process(cv::Mat const& new_image, int image_id, double image_time, bool& new_keyframe)
{
  cv::cvtColor(new_image, current_img_, cv::COLOR_BGR2GRAY);
  current_img_id_ = image_id;
  current_img_time_ = image_time;

  if(use_rotation_compensated_parallax_) update_relative_rotation();
  process_current_image();

  new_keyframe = new_keyframe_declared_;
  return feat_manager_->feats;
}

void FeatureTracker::clear_all_features()
{
  feat_manager_->clear_all();
}

// get approximate rotation between previous keyframe and current image
// for computing rotation compensated parallax
void FeatureTracker::update_relative_rotation()
{
  R_rel_ = R_bc_ * gyro_integrator_->update(current_img_time_) * R_bc_.inverse();
}

void FeatureTracker::process_current_image()
{
  if(feat_manager_->feats->size()) find_optic_flow();

  std::vector<cv::Point2f> feat_new, feat_new_norm;
  find_new_features(feat_new);
  if(feat_new.size()) cv::undistortPoints(feat_new, feat_new_norm, K_, D_);

  feat_manager_->new_features(feat_new, feat_new_norm, current_img_id_);

  update_keyframes();

  prev_img_ = current_img_.clone();
  feat_manager_->clear_inactive();
}

void FeatureTracker::find_new_features(std::vector<cv::Point2f>& feat_new)
{
  cv::Mat img_crop, mask_crop;
  cv::Mat mask = cv::Mat::ones(current_img_.size(), CV_8UC1)*255;
  create_mask(mask);
  std::vector<int> visible_pts_per_part;
  find_visible_pts_per_part(visible_pts_per_part);

  std::vector<cv::Point2f> feat_part;
  int max_corners;
  for(int i = 0; i < part_rects_.size(); i++)
  {
    img_crop = current_img_(part_rects_[i]);
    mask_crop = mask(part_rects_[i]);
    max_corners = gf2t_max_corners_ - visible_pts_per_part[i];
    if(max_corners <= 0)
      continue;

    cv::goodFeaturesToTrack(img_crop, feat_part, max_corners, gf2t_quality_level_, gf2t_min_distance_, mask_crop);

    if(!feat_part.size()) continue;

    cv::cornerSubPix(img_crop, feat_part, subpix_win_size_, cv::Size(-1,-1), subpix_criteria_);

    for(int j = 0; j < feat_part.size(); ++j) feat_part[j] = feat_part[j] + cv::Point2f(part_rects_[i].x, part_rects_[i].y);

    feat_new.insert(feat_new.end(), feat_part.begin(), feat_part.end());
    feat_part.clear();
  }
}

void FeatureTracker::find_optic_flow()
{
  std::vector<cv::Point2f> feat_visible;
  std::vector<int> feat_ids;
  feat_manager_->get_visible_feats(feat_visible, feat_ids);

  if(!feat_visible.size()) return;

  std::vector<cv::Point2f> feat_next, feat_next_norm;
  std::vector<uchar> status;
  std::vector<float> err;
  cv::calcOpticalFlowPyrLK(prev_img_, current_img_, feat_visible, feat_next, status, err, klt_win_size_, klt_max_level_, klt_criteria_);
  filter_matches(feat_visible, feat_next, status);

  if(feat_next.size()) cv::undistortPoints(feat_next, feat_next_norm, K_, D_);

  for(int i = 0; i < status.size(); i++)
  {
    if(status[i] == 1)
      feat_manager_->update_single(feat_ids[i], feat_next[i], feat_next_norm[i], current_img_id_);
  }

  feat_manager_->label_not_visible(feat_ids, status);
  visible_pts_.clear();
  filter_points(feat_next, status, visible_pts_);
}

void FeatureTracker::filter_matches(std::vector<cv::Point2f> const& p_i, std::vector<cv::Point2f> const& p_j, std::vector<uchar>& status)
{
  std::vector<cv::Point2f> good_feats_prev, good_feats_next;
  filter_points(p_i, status, good_feats_prev);
  filter_points(p_j, status, good_feats_next);

  // fundamental matrix estimation requires at least 8 points
  if(good_feats_prev.size() < 8) return; 

  std::vector<uchar> good;
  cv::findFundamentalMat(good_feats_prev, good_feats_next, cv::FM_RANSAC, 1.0, 0.99, good); //play with accuracy here

  int good_it = 0;
  for(int i = 0; i < status.size(); i++)
  {
    if(status[i] != 1) continue;
    
    if(!good[good_it++]) status[i] = 0;
  }
}

void FeatureTracker::filter_points(std::vector<cv::Point2f> const& pts, std::vector<uchar> const& status, std::vector<cv::Point2f>& filtered_pts)
{
  for(int i = 0; i < status.size(); i++)
  {
    if(status[i] == 1) filtered_pts.push_back(pts[i]);
  }
}

void FeatureTracker::update_keyframes()
{
  std::vector<int> match_ids;
  std::vector<cv::Point2f> match_i, match_j;

  new_keyframe_declared_ = feat_manager_->manage_keyframes(match_ids, match_i, match_j, use_rotation_compensated_parallax_ ? &R_rel_ : nullptr);

  if(new_keyframe_declared_)
  {
    // need to filter from old keyframe to new keyframe whenever new keyframe is declared
    if(match_ids.size())
    {
      std::vector<uchar> status(match_i.size());
      std::fill(status.begin(), status.end(), 1);
      
      filter_matches(match_i, match_j, status);
      feat_manager_->label_not_visible(match_ids, status);
    }

    // reset relative rotation to identity, restart gyro integration
    if(use_rotation_compensated_parallax_)
    {
      R_rel_ = lie_groups::SO3d();
      gyro_integrator_->reset(current_img_time_);
    }
  }
}

void FeatureTracker::create_mask(cv::Mat& mask)
{
  for(auto const& pt : visible_pts_) cv::circle(mask, pt, gf2t_min_distance_, cv::Scalar(0), -1);
}

void FeatureTracker::find_visible_pts_per_part(std::vector<int>& visible_pts_per_part)
{
  for(int i = 0; i < part_rects_.size(); i++)
  {
    visible_pts_per_part.push_back(0);
  }

  int x_bin, y_bin;
  int w = part_rects_[0].width;
  int h = part_rects_[0].height;
  for(auto const& pt : visible_pts_)
  {
    x_bin = pt.x / w;
    y_bin = pt.y / h;
    visible_pts_per_part[y_bin*num_part_w_ + x_bin]++;
  }
}

void FeatureTracker::get_draw_image(cv::Mat& draw_img)
{
  feat_manager_->get_draw_img(current_img_, current_img_id_, draw_img);
}

} //namespace vision