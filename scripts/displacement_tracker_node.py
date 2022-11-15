import cv2
import rospy
import numpy as np
from collections import deque
from ttc_object_avoidance.msg import TrackedFeats, TrackedFeatsWDis, FeatAndDisplacement
from sensor_msgs.msg import Image

class DisplacementTrackerNode:
    feature_disp_publisher = None

    trackedFeaturesPath = None


    def __init__(self) -> None:
        self.trackedIDs = set()
        self.trackedFeaturesPath = dict()
        rospy.Subscriber("tracked_features", TrackedFeats, self.features_callback, queue_size=1)
        self.feature_disp_publisher = rospy.Publisher("tracked_disp", TrackedFeatsWDis, queue_size=1)

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

            path.append(currentFeature.pts[-1])
            while len(path) > 2:
                path.popleft()
            
            self.trackedFeaturesPath[currentID] = path


        for oldID in self.trackedIDs:
            if oldID not in newTrackedIdDs:
                self.trackedFeaturesPath.pop(oldID)
        
        self.trackedIDs = newTrackedIdDs

        # run through the features and calculate displacements. Only include a feature in the result if it has a displacement
        features = []
        

        for featID in self.trackedFeaturesPath:
            path = self.trackedFeaturesPath[featID]

            # check if we can calculate the displacement
            if len(path) < 2:
                continue

            disx = path[-1].x - path[-2].x
            disy = path[-1].y - path[-2].y

            feat = FeatAndDisplacement()
            feat.feat_id=featID
            feat.pt = path[-1]
            feat.displacement.x = disx
            feat.displacement.y = disy
            feat.displacement.z = 0
            features.append(feat)
        
        featsMsg = TrackedFeatsWDis()
        featsMsg.header = feats.header
        featsMsg.new_keyframe = feats.new_keyframe
        featsMsg.feats = features
        self.feature_disp_publisher.publish(featsMsg)