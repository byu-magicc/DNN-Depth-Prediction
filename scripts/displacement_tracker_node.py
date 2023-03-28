#!/usr/bin/env python3

import cv2
import rospy
import numpy as np
from collections import deque
from ttc_object_avoidance.msg import TrackedFeats, TrackedFeatsWDis, FeatAndDisplacement
from sensor_msgs.msg import Image

class DisplacementTrackerNode:
    feature_disp_publisher = None

    trackedFeaturesPath = None

    pastTimeS = None
    pastTimeNs = None

    # This part calibrates the pixels according to AirSim's camera characteristics
    # Remove if using normalized pixels from the message (see below)
    fx = 465.6
    fy = fx
    cx = 320
    cy = 240

    def calibrate_pixels(self, pos):
        new_pos = np.reshape(pos, (2, -1))
        uncalibrated = np.ones((3,1))
        uncalibrated[0:2] = new_pos

        calibrated = self.cam_inv @ uncalibrated
        return np.reshape(calibrated[0:2], pos.shape)
    # END part

    def __init__(self) -> None:
        self.trackedIDs = set()
        self.trackedFeaturesPath = dict()
        ns = rospy.get_namespace()
        self.fx = rospy.get_param(ns + "projection_parameters/fx", self.fx)
        self.fy = rospy.get_param(ns + "projection_parameters/fy", self.fy)
        self.cx = rospy.get_param(ns + "projection_parameters/cx", self.cx)
        self.cy = rospy.get_param(ns + "projection_parameters/cy", self.cy) # TODO: maybe let Jake's stuff handle the calibration with the actual camera.
        camera_intrinsics = np.array([[self.fx, 0, self.cx],
                                [0, self.fy, self.cy],
                                [0, 0, 1]])

        self.cam_inv = np.linalg.inv(camera_intrinsics)
        rospy.Subscriber("tracked_features", TrackedFeats, self.features_callback, queue_size=1)
        publish_topic = rospy.get_param(ns + "/feat_velocity_topic", "tracked_disp")
        self.feature_disp_publisher = rospy.Publisher(publish_topic, TrackedFeatsWDis, queue_size=1)

        

    def run(self):
        rospy.spin()

    def features_callback(self, feats):
        newTrackedIdDs = set()
        

        for currentFeature in feats.feats:
            currentID = currentFeature.feat_id
            newTrackedIdDs.add(currentID)

            if currentID not in self.trackedFeaturesPath:
                self.trackedFeaturesPath[currentID] = deque()
            
            path = self.trackedFeaturesPath[currentID]

            path.append(currentFeature.pts[-1]) #change to norm_pts if you want to use normalized points from tracker
            while len(path) > 2:
                path.popleft()
            
            self.trackedFeaturesPath[currentID] = path


        for oldID in self.trackedIDs:
            if oldID not in newTrackedIdDs:
                self.trackedFeaturesPath.pop(oldID)
        
        self.trackedIDs = newTrackedIdDs

        # run through the features and calculate displacements. Only include a feature in the result if it has a displacement
        features = []
        currentTimeS = feats.header.stamp.secs
        currentTimeNs = feats.header.stamp.nsecs

        for featID in self.trackedFeaturesPath:
            path = self.trackedFeaturesPath[featID]

            # check if we can calculate the displacement
            if len(path) < 2 or self.pastTimeS is None:
                continue

            current_feat_location = np.array([path[-1].x, path[-1].y])
            previous_feat_location = np.array([path[-2].x, path[-2].y])
            calibrated_current_location = self.calibrate_pixels(current_feat_location)
            calibrated_previous_location = self.calibrate_pixels(previous_feat_location)
            calibrated_displacement = calibrated_current_location - calibrated_previous_location
            # disx = path[-1].x - path[-2].x
            # disy = path[-1].y - path[-2].y

            # Normalize the points according to AirSim's camera characteristics

            feat = FeatAndDisplacement()
            feat.feat_id=featID
            feat.pt.x = calibrated_current_location[0]
            feat.pt.y = calibrated_current_location[1]

            delt = (currentTimeS - self.pastTimeS) + (currentTimeNs - self.pastTimeNs)/1000000000.

            feat.displacement.x = calibrated_displacement[0] / delt
            feat.displacement.y = calibrated_displacement[1] / delt
            feat.displacement.z = 0
            features.append(feat)
        
        featsMsg = TrackedFeatsWDis()
        featsMsg.header = feats.header
        featsMsg.new_keyframe = feats.new_keyframe
        featsMsg.feats = features
        self.feature_disp_publisher.publish(featsMsg)
        self.pastTimeS = currentTimeS
        self.pastTimeNs = currentTimeNs

if __name__=="__main__":
    rospy.init_node("displacement_tracker")
    ros_node = DisplacementTrackerNode()
    ros_node.run()