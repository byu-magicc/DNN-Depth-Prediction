#!/usr/bin/env python3

import cv2
import csv
import rospy
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from ttc_object_avoidance.msg import TrackedFeatsWDis

class AirSimRecordingReaderNode:
    image_pub = None
    record_lines = []
    line_num = 0
    featfilename = ""

    dirToRead = "/home/james/Documents/AirSim/supercomputer_recording/"

    def __init__(self) -> None:
        self.image_pub = rospy.Publisher("camera_image", Image, queue_size=1)
        rospy.Subscriber("tracked_disp", TrackedFeatsWDis, self.feature_callback, queue_size=1)
        try:
            recording_file = open(self.dirToRead + "airsim_rec.txt","r")
            data_reader = csv.DictReader(recording_file, delimiter="\t")
            for line in data_reader:
                self.record_lines.append(line)
            recording_file.close()

        except:
            print("Could not open recording file. Make sure the path is correct")
            #TODO: Kill the nodes

    # Function called to publish the next frame of the recording
    def publish_data(self) -> None:
        #dimensions of the image for the algorithm
        dim=(640,480)

        bridge = CvBridge()
        line = self.record_lines[self.line_num]

        # get the filenames from the line
        both_images = line["ImageFile"]

        # get the color image filename
        color_filename = both_images.split(";")[0]

        # save the base of the filename for later
        self.featfilename = color_filename.split(".")[0]

        # read the frame from the file
        img = cv2.imread(self.dirToRead + "images/" + color_filename, cv2.IMREAD_COLOR)

        # resize it to the proper scale
        resized_img = cv2.resize(img, dim)

        # publish it to the people
        self.image_pub.publish(bridge.cv2_to_imgmsg(resized_img, encoding="bgr8"))


        self.prev_num = self.line_num
    
    # Function to summarize the features and their velocities and add them to the file
    def feature_callback(self, feats) -> None:
        if len(feats.feats) < 4:
            rospy.loginfo("Insufficient number of features recieved to get depth data")
            self.line_num += 1
            return
        with open(self.dirToRead + "feat_data/" + self.featfilename + ".csv", "w", newline="") as featDataFile:
            dataWriter = csv.writer(featDataFile)
            header = ["pos_x", "pos_y", "vel_x", "vel_y"]
            dataWriter.writerow(header)
            del_t = int(self.record_lines[self.line_num]["TimeStamp"]) - int(self.record_lines[self.line_num-1]["TimeStamp"])
            del_t /= 1000

            for feat in feats.feats:
                row = [feat.pt.x, feat.pt.y, feat.displacement.x/del_t, feat.displacement.y/del_t]
                dataWriter.writerow(row)
        self.line_num += 1

if __name__ == "__main__":
    rospy.init_node("airsim_recording_reader_node", anonymous=True)
    ros_node = AirSimRecordingReaderNode()
    rate = rospy.Rate(5)

    while not rospy.is_shutdown():
        ros_node.publish_data()
        rate.sleep()