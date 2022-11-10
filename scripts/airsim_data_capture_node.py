import cv2
import airsim
import rospy
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import pprint

class AirsimDataCaptureNode:
    image_pub = None
    client = None
    latest_lv = None
    latest_av = None

    def __init__(self) -> None:
        self.image_pub = rospy.Publisher("camera_image", Image, queue_size=1)
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

    def capture_data(self) -> None:
        bridge = CvBridge()
        self.client.simPause(True)
        # info = self.client.simGetCameraInfo("front-center")
        # print("info: %s" % pprint.pformat(info))
        # responses = self.client.simGetImages([
        #     airsim.ImageRequest("0", airsim.ImageType.Scene),
        #     airsim.ImageRequest("0", airsim.ImageType.DepthVis, True)
        # ])
        state = self.client.getMultirotorState()
        self.client.simPause(False)

        print("state: %s" % pprint.pformat(state))

        self.latest_av = state.kinematics_estimated.angular_velocity
        self.latest_lv = state.kinematics_estimated.linear_velocity

        # scene = responses[0]
        # self.latest_depth = responses[1]

        # scene_img = cv2.imdecode(airsim.string_to_uint8_array(scene), cv2.IMREAD_UNCHANGED)

        # resized_scene = cv2.resize(scene_img, (640, 480))
        # self.image_pub.publish(bridge.cv2_to_imgmsg(resized_scene, encoding="bgr8"))






    

if __name__ == "__main__":
    rospy.init_node("airsim_data_capture_node", anonymous=True)
    ros_node = AirsimDataCaptureNode()
    rate = rospy.Rate(30)

    while not rospy.is_shutdown():
        ros_node.capture_data()
        rate.sleep()