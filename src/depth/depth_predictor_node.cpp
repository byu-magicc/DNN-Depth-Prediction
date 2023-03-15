#include "depth/depth_predictor_node.h"

namespace depth
{

DepthPredictorNode::DepthPredictorNode(){
    Eigen::Matrix3f body_to_cam;
    body_to_cam << 0, 1, 0,
                   0, 0, 1,
                   1, 0, 0;
    Eigen::Vector3f camera_offset(0.0, 0.0, 0.3);
    depth_predictor_ = DepthPredictor(body_to_cam, camera_offset);
}

}



int main(int argc, char **argv) {
    depth::DepthPredictorNode *depth_tracker = new depth::DepthPredictorNode();
    return 0;
}