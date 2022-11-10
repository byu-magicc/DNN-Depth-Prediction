import cv2
import csv
import rospy
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

class AirSimRecordingReaderNode:
    image_pub = None
    record_lines = []
    line_num = 0

    def __init__(self) -> None:
        self.image_pub = rospy.Publisher("camera_image", Image, queue_size=1)
        try:
            recording_file = open("airsim_rec.txt","r")
            data_reader = csv.DictReader(recording_file, delimiter="\t")
            for line in data_reader:
                self.record_lines.append(line)


        except:
            print("Could not open recording file. Make the script is run inside an AirSim recording folder")
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

        # read the frame from the file
        img = cv2.imread("images/" + color_filename, cv2.IMREAD_COLOR)

        # resize it to the proper scale
        resized_img = cv2.resize(img, dim)

        # publish it to the people
        self.image_pub.publish(bridge.cv2_to_imgmsg(resized_img, encoding="bgr8"))


        #self.line_num += 1
    
    # Function to summarize the features and their velocities and add them to the file
    def feature_callback(self, feats) -> None:
        if len(feats.feats) < 4:
            rospy.loginfo("Insufficient number of features recieved to get depth data")
            return
        