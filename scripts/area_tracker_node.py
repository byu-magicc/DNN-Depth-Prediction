#!/usr/bin/env python3

import cv2
import rospy
import numpy as np
from ttc_object_avoidance.msg import TrackedFeats
from geometry_msgs.msg import Point
from scipy.spatial import Delaunay

class AreaTrackerNode:
    triangles_found = False
    trackedIDs = None
    delaunay_simplices = None
    delaunay_simplices_ids = None
    mapIdsToIndecies = None

    currentFeaturePositions = np.zeros((1000, 2), dtype=int)
    currentFeatureIDs = np.zeros((1000), dtype=int)
    currentNumPoints = 0

    def __init__(self) -> None:
        self.trackedIDs = set()
        self.mapIdsToIndecies = dict()
        rospy.Subscriber("tracked_features", TrackedFeats, self.features_callback, queue_size=1)

    def run(self):
        rospy.spin()

    def features_callback(self, feats):
        self.trackedIDs.clear()
        i = -1
        for feat in feats.feats:
            i += 1
            currentID = feat.feat_id
            self.currentFeatureIDs[i] = currentID
            self.mapIdsToIndecies[currentID] = i
            featurePos = feat.norm_pts[-1]
            self.currentFeaturePositions[i] = np.array([featurePos.x, featurePos.y])
            self.trackedIDs.add(currentID)

        if self.triangles_found:
            newSimplicesIndecies = np.zeros_like(self.delaunay_simplices_ids, dtype=int)
            newSimplicesIDs = np.zeros_like(self.delaunay_simplices_ids, dtype=int)
            i = 0
            for tri in self.delaunay_simplices_ids:
                valid_simplex = True
                for j in range(3):
                    if tri[j] not in self.trackedIDs:
                        valid_simplex = False
                        break
                    newSimplicesIndecies[i, j] = self.mapIdsToIndecies[tri[j]]
                    newSimplicesIDs[i,j] = tri[j]
                if valid_simplex:
                    i += 1
            self.delaunay_simplices = newSimplicesIndecies[:i + 1]
            self.delaunay_simplices_ids = newSimplicesIDs[:i + 1]

        #if not self.delaunay_calculated or self.delaunay_simplices.shape[0] < 200:
        tri = Delaunay(self.currentFeaturePositions)
        self.delaunay_simplices = tri.simplices
        self.delaunay_simplices_ids = self.currentFeatureIDs[tri.simplices]
        self.triangles_found = True
        #self.previous_areas = self.calculateAreas(self.delaunay_simplices_ids, self.delaunay_simplices, self.currentFeaturePositions)


if __name__=='__main__':
    rospy.init_node('area_tracker', anonymous=True)
    try:
        ros_node = AreaTrackerNode()
        ros_node.run()
    except rospy.ROSInterruptException:
        pass