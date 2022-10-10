#pragma once

#include <vector>
#include <map>

#include <opencv2/opencv.hpp>

#include "feature_manager.h"

namespace vision 
{

class DelaunayAreaTracker {
private:
    std::map<std::string, double> areas;

    void updateAreas(vector<Feature> features);
    void get_draw_img(cv::Mat const& img, cv::Mat& draw_img);
}
}