#pragma once

#include <vector>
#include <ros/ros.h>
#include <Eigen/Dense>

#include "depth/depth_predictor.h"

namespace depth
{

class DepthPredictorNode
{
public:
    DepthPredictorNode();

private:
    DepthPredictor depth_predictor_;
};
}