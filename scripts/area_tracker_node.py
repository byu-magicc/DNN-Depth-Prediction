#!/usr/bin/env python3

import queue
from tokenize import Double
import cv2
import rospy
import numpy as np
from ttc_object_avoidance.msg import TrackedFeats
from sensor_msgs.msg import Image
from scipy.spatial import Delaunay
from cv_bridge import CvBridge, CvBridgeError

# This is a ROS node designed to take images and track features on
# them. It takes these features and groups them together into non-
# overlapping triangles. It then tracks the area of these triangles
# and calculates a TTC using the change in area. This idea was
# eventually scrapped because the loss of feature tracking on a 
# feature breaks the area calculation on any triangle it was a part
# of.

class AreaTrackerNode:
    triangles_found = False
    trackedIDs = None
    delaunay_simplices = None
    delaunay_simplices_ids = None
    mapIdsToIndecies = None

    currentFeaturePositions = np.zeros((1000, 2), dtype=float)
    currentFeatureIDs = np.zeros((1000), dtype=int)
    currentNumPoints = 0

    currentAreas = None
    previousTTC = None

    previousStamp = 0

    idPadding = 10

    # printedAlready = False
    draw_areas = False

    def __init__(self) -> None:
        self.trackedIDs = set()
        self.mapIdsToIndecies = dict()
        self.previousTTC = dict()
        rospy.Subscriber("tracked_features", TrackedFeats, self.features_callback, queue_size=1)
        ns = rospy.get_namespace()
        self.draw_areas = rospy.get_param(ns + 'pub_draw_feats')

        if self.draw_areas:
            rospy.Subscriber("features_image", Image, self.image_callback,queue_size=1)
            self.image_pub = rospy.Publisher("area_image", Image, queue_size=1)

    def run(self):
        rospy.spin()

    def features_callback(self, feats):
        if len(feats.feats) < 4:
            rospy.loginfo("Insufficient number of features received")
            return
        self.trackedIDs.clear()
        currentStamp = feats.header.stamp
        i = -1
        for feat in feats.feats:
            i += 1
            currentID = feat.feat_id
            self.currentFeatureIDs[i] = currentID
            self.mapIdsToIndecies[currentID] = i
            featurePos = feat.pts[-1]
            self.currentFeaturePositions[i, 0] = featurePos.x
            self.currentFeaturePositions[i, 1] = featurePos.y
            self.trackedIDs.add(currentID)

        self.currentNumPoints = i + 1
        lastIndex = i

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
            self.delaunay_simplices = newSimplicesIndecies[:i]
            self.delaunay_simplices_ids = newSimplicesIDs[:i]
            
            #
            self.previousTTC.clear()
            newAreas = self.calculateAreas(self.delaunay_simplices_ids, self.delaunay_simplices, self.currentFeaturePositions)
            # rospy.loginfo("New Keys:" + str(newAreas.keys()))
            for id_str, newArea in newAreas.items():
                oldArea = self.currentAreas[id_str]
                ttc = (currentStamp - self.previousStamp).nsecs*1e-9 * newArea/(newArea - oldArea)
                self.previousTTC[id_str] = ttc

        self.previousStamp = currentStamp
        tri = Delaunay(self.currentFeaturePositions[:lastIndex])
        self.delaunay_simplices = tri.simplices
        self.delaunay_simplices_ids = self.currentFeatureIDs[tri.simplices]
        self.triangles_found = True
        self.currentAreas = self.calculateAreas(self.delaunay_simplices_ids, self.delaunay_simplices, self.currentFeaturePositions)
        # rospy.loginfo("Current Keys:" + str(self.currentAreas.keys()))

    def image_callback(self, img_msg):
        bridge = CvBridge()
        try:
            cv_image = bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
        overlay = cv_image.copy()
        previousTTCItems = self.previousTTC.items()
        for id_str, ttc in previousTTCItems:
            index1 = self.mapIdsToIndecies[int(id_str[0:self.idPadding])]
            index2 = self.mapIdsToIndecies[int(id_str[self.idPadding:self.idPadding*2])]
            index3 = self.mapIdsToIndecies[int(id_str[self.idPadding*2:])]
            simplex = np.array([index1, index2, index3])
            color = (0, 128, 0)
            if ttc < 0:
                pass
            elif ttc < 3:
                color = (0, 0, 128)
            elif ttc < 10:
                color = (0, 128, 128)
            # rospy.loginfo("Simplex: " + str(simplex) + ", positions: " + str(self.currentFeaturePositions[simplex].astype(int)))
            cv2.drawContours(overlay, [self.currentFeaturePositions[simplex].astype(int)], -1, color, -1)
        alpha = 0.5
        img = cv2.addWeighted(overlay, alpha, cv_image, 1-alpha, 0)
        self.image_pub.publish(bridge.cv2_to_imgmsg(img,encoding="bgr8"))

    def calculateAreas(self, simplicesIDs, simplices, points):
        areas = dict()

        trianglePoints = points[simplices]
        areasArray = 1./2. * np.abs(trianglePoints[:, 0, 0] * (trianglePoints[:, 1, 1] - trianglePoints[:, 2, 1]) 
                               + trianglePoints[:, 1, 0] * (trianglePoints[:, 2, 1] - trianglePoints[:, 0, 1])
                               + trianglePoints[:, 2, 0] * (trianglePoints[:, 0, 1] - trianglePoints[:, 1, 1]))
        for i in range(areasArray.shape[0]):
            pointIDs = np.sort(simplicesIDs[i])
            triangleID = str(pointIDs[0]).zfill(self.idPadding) + str(pointIDs[1]).zfill(self.idPadding) + str(pointIDs[2]).zfill(self.idPadding)
            areas[triangleID] = areasArray[i]
        
        return areas

if __name__=='__main__':
    rospy.init_node('area_tracker', anonymous=True)
    try:
        ros_node = AreaTrackerNode()
        ros_node.run()
    except rospy.ROSInterruptException:
        pass