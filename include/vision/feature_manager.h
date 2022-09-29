#pragma once

#include <vector>
#include <map>
#include <utility>

#include <opencv2/opencv.hpp>

#include "lie_groups/so3.hpp"

namespace vision
{

struct Feature
{
  Feature()
  {}

  Feature(int id, const cv::Scalar col) : 
    feat_id(id), draw_color(col), visible(true), active(false)
  {}

  ~Feature()
  {
    image_ids.clear();
    pts.clear();
    norm_pts.clear();
  }

  void update(cv::Point2f pt, cv::Point2f norm_pt, int img_id)
  {
    image_ids.push_back(img_id);
    pts.push_back(pt);
    norm_pts.push_back(norm_pt);
  }

  void keyframe_update(int oldest_keyframe)
  {
    keyframe_pts.push_back(std::make_pair(image_ids.back(), std::make_pair(pts.back(), norm_pts.back())));
    while(keyframe_pts[0].first < oldest_keyframe) keyframe_pts.erase(keyframe_pts.begin());
  }

  cv::Point2f get_last_seen()
  {
    return pts.back();
  }

  std::vector<int> image_ids;
  std::vector<cv::Point2f> pts;
  std::vector<cv::Point2f> norm_pts;
  int feat_id;
  cv::Scalar draw_color;
  bool visible;
  bool active; // active if keyframe_pts.size() > minimum_keyframes_for_inclusion

  std::vector<std::pair<int, std::pair<cv::Point2f, cv::Point2f>>> keyframe_pts; //key: keyframe img id, data: <pt, norm_pt>
};


struct FeatureManager
{
  FeatureManager() : 
    current_id(0), feats(new std::map<int, Feature>())
  {}
  
  FeatureManager(cv::Mat K, int keyframe_window_length, double keyframe_parallax_threshold, int keyframe_minimum_features, int minimum_keyframes_for_inclusion) : 
    K(K), K_inv(K.inv()), keyframe_window_length(keyframe_window_length), keyframe_parallax_threshold(keyframe_parallax_threshold),
    keyframe_minimum_features(keyframe_minimum_features), minimum_keyframes_for_inclusion{minimum_keyframes_for_inclusion}, current_id(0),
    feats(new std::map<int, Feature>())
  {}

  ~FeatureManager()
  {
    clear_all();
  }

  void new_features(std::vector<cv::Point2f> pts_new, std::vector<cv::Point2f> pts_new_norm, int img_id)
  {   
    current_img_id = img_id;

    for(int i = 0; i < pts_new.size(); i++)
    {
      Feature new_feat(current_id, cv::Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255)));
      new_feat.update(pts_new[i], pts_new_norm[i], img_id);
      feats->insert({current_id++, new_feat});
    }
  }

  void get_visible_feats(std::vector<cv::Point2f>& feat_visible, std::vector<int>& feat_ids)
  {
    Feature feat;
    for(auto const& map_feat : *feats)
    {   
      feat = map_feat.second;
      if(!feat.visible)
        continue;
      
      feat_visible.push_back(feat.pts.back());
      feat_ids.push_back(feat.feat_id);
    }
  }

  void update_single(int feat_id, cv::Point2f new_pt, cv::Point2f new_pt_norm, int img_id)
  {
    std::map<int, Feature>::iterator it = feats->find(feat_id);
    it->second.update(new_pt, new_pt_norm, img_id);
  }

  void label_not_visible(std::vector<int> feat_ids, std::vector<uchar> found)
  {
    std::map<int, Feature>::iterator it;
    for(int i = 0; i < feat_ids.size(); i++)
    {
      if(found[i] != 1)
      {
        it = feats->find(feat_ids[i]);
        it->second.visible = false;
        inactive_list.push_back(feat_ids[i]);
      }
    }
  }

  bool manage_keyframes(std::vector<int>& keyframe_pair_id, std::vector<cv::Point2f>& keyframe_pair_i, std::vector<cv::Point2f>& keyframe_pair_j, lie_groups::SO3d* R_ij = nullptr)
  {
    double average_parallax;
    int num_keyframe_pts_active;
    find_average_parallax(average_parallax, num_keyframe_pts_active, R_ij);

    if(average_parallax > keyframe_parallax_threshold || num_keyframe_pts_active < keyframe_minimum_features)
    {
      // cout << "new keyframe! Parallax: " << average_parallax << " Points: " << num_keyframe_pts_active << endl;
      // add new keyframe, delete oldest keyframe
      keyframes.push_back(current_img_id);
      if(keyframes.size() > keyframe_window_length) keyframes.erase(keyframes.begin());

      for(auto &current_feat : *feats)
      {
        if(current_feat.second.visible)
        {
          current_feat.second.keyframe_update(keyframes[0]);
          // if a feature exists in the last two keyframes, add it to output vectors to determine if it's a good match
          if(current_feat.second.keyframe_pts.size() >= 2)
          {
            keyframe_pair_id.push_back(current_feat.second.feat_id);
            keyframe_pair_i.push_back(current_feat.second.keyframe_pts[current_feat.second.keyframe_pts.size()-2].second.first);
            keyframe_pair_j.push_back(current_feat.second.keyframe_pts[current_feat.second.keyframe_pts.size()-1].second.first);
          }
        }

        // label active or inactive
        if(current_feat.second.keyframe_pts.size() >= minimum_keyframes_for_inclusion) current_feat.second.active = true;
        else current_feat.second.active = false;
      }

      return true;
    }
    else
    {
      return false;
    }
  }

  void find_average_parallax(double &average_parallax, int &num_keyframe_pts_active, lie_groups::SO3d* R_ij = nullptr)
  {
    double sum_parallax = 0.0;
    int num_feats = 0;
    if(keyframes.size())
    {
      cv::Mat R_ji_cv;
      if(R_ij != nullptr) cv::eigen2cv((*R_ij).inverse().matrix(), R_ji_cv);
      for(auto const& feat : *feats)
      {
        // only look at features that are visible and were present in the last keyframe
        if(feat.second.visible && feat.second.keyframe_pts.size())
        {
          cv::Mat pt_i(3, 1, CV_64F);
          pt_i.at<double>(0,0) = feat.second.keyframe_pts.back().second.first.x;
          pt_i.at<double>(1,0) = feat.second.keyframe_pts.back().second.first.y;
          pt_i.at<double>(2,0) = 1.0;

          cv::Mat pt_j(3, 1, CV_64F);
          pt_j.at<double>(0,0) = feat.second.pts.back().x;
          pt_j.at<double>(1,0) = feat.second.pts.back().y;
          pt_j.at<double>(2,0) = 1.0;

          sum_parallax += calculate_parallax(pt_i, pt_j, (R_ij == nullptr) ? nullptr : &R_ji_cv);
          num_feats++;
        }
      }
    }

    if(num_feats > 0)
      average_parallax = sum_parallax/(double) num_feats;
    else
      average_parallax = 0.0;

    num_keyframe_pts_active = num_feats;
  }

  // calculate parallax given an unnormalized image location in the i and j frame, and if R_ij given, compensate for rotation
  double calculate_parallax(cv::Mat pt_i, cv::Mat pt_j, cv::Mat* R_ji = nullptr)
  {
    if(R_ji == nullptr)
      return cv::norm(pt_i, pt_j, cv::NORM_L2);
    else
    {
      cv::Mat pt_j_comp(3, 1, CV_64F);
      cv::Mat pt_j_bar(3, 1, CV_64F);
      pt_j_bar = K_inv * pt_j;
      pt_j_comp = K * (*R_ji) * (pt_j_bar/cv::norm(pt_j_bar, cv::NORM_L2));
      pt_j_comp = (1.0 / pt_j_comp.at<double>(2,0)) * pt_j_comp;

      return cv::norm(pt_i, pt_j_comp, cv::NORM_L2);
    }
  }

  void get_draw_img(cv::Mat const& img, int img_id, cv::Mat& draw_img)
  {
    cv::cvtColor(img, draw_img, cv::COLOR_GRAY2BGR);

    std::vector<cv::Point2f> pts_visible;
    std::vector<cv::Scalar> colors;
    std::vector<bool> active;
    for(auto const& map_feat : *feats)
    {
      if(map_feat.second.visible)
      {
        pts_visible.push_back(map_feat.second.pts.back());
        colors.push_back(map_feat.second.draw_color);
        active.push_back(map_feat.second.active);
      }
    }
    for(int i = 0; i < pts_visible.size(); i++)
    {
      cv::circle(draw_img, pts_visible[i], 4, colors[i], 2);
      if(active[i])
        cv::rectangle(draw_img, cv::Point2f(pts_visible[i].x-6, pts_visible[i].y-6), cv::Point2f(pts_visible[i].x+6, pts_visible[i].y+6), colors[i], 2);
    }
  }

  void clear_inactive()
  {
    std::map<int, Feature>::iterator it = feats->begin();
    while(it != feats->end())
    {
      // delete feature if it is no longer in view and does not have enough keyframes
      if(!it->second.visible && !it->second.active)
        it = feats->erase(it);
      else
        it++;
    }
  }

  void clear_all()
  {
    feats->clear();
  }

  std::shared_ptr<std::map<int, Feature>> feats;
  int current_id;
  int current_img_id;
  cv::Mat K;
  cv::Mat K_inv;
  cv::RNG rng;
  std::vector<int> inactive_list;

  std::vector<int> keyframes; // image ID of keyframes;
  double keyframe_parallax_threshold; // average parallax required before new keyframe is declared
  double keyframe_window_length; // number of keyframes in sliding window
  int keyframe_minimum_features; // minimum feature number required before new keyframe is declared
  int minimum_keyframes_for_inclusion; // minimum number of keframes required to include feature in estimation
};

} // namespace vision