#!/usr/bin/env python3
# Description:
# - Subscribes to real-time streaming video from your built-in webcam.
#
# Author:
# - Addison Sears-Collins
# - https://automaticaddison.com
 
# Import the necessary libraries
import cv2 # OpenCV library
import numpy as np
import csv

class NaiveFeatureTracker:
    record_lines = []
    line_num = 0
    featfilename = ""

    dirToRead = "/home/james/Documents/AirSim/supercomputer_recording/"

    # Show feature tracks?
    SHOW_TRACKS = False

    SHOW_IMAGE = True

    # Parameters for the goodFeaturesToTrack function
    feature_params = dict(maxCorners=0,
                        qualityLevel=0.2,
                        minDistance=10,
                        blockSize=3)

    # Parameters for the LK Optic Flow function
    lk_params=dict(winSize=(15,15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Line colors
    colors = np.array([[0, 0, 63], [0, 255, 0]])

    mask = None

    # List of features to track
    next_feats = None
    old_gray = None

    displacements = None

    f = 465.6
    camera_intrinsics = np.array([[f, 0, 320],
                                [0, f, 240],
                                [0, 0, 1]])

    cam_inv = np.linalg.inv(camera_intrinsics)

    def calibrate_pixels(self, pos):
        new_pos = np.reshape(pos, (2,-1))
        uncalibrated = np.ones((3,new_pos.shape[1]))
        uncalibrated[0:2] = pos

        calibrated = self.cam_inv @ uncalibrated
        return np.reshape(calibrated[0:2], pos.shape)

    def __init__(self) -> None:
        try:
            recording_file = open(self.dirToRead + "airsim_rec.txt","r")
            data_reader = csv.DictReader(recording_file, delimiter="\t")
            for line in data_reader:
                self.record_lines.append(line)
            recording_file.close()

        except:
            print("Could not open recording file. Make sure the path is correct")
            #TODO: Kill the nodes

    def track_features(self, current_frame):

        # Convert the frame to grayscale
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Check if we have any frames (or later if we have enough frames)
        if self.next_feats is None:
            self.next_feats = cv2.goodFeaturesToTrack(current_gray, mask=None, **self.feature_params)

            # Keep track of old gray frame
            self.old_gray = current_gray.copy()

            # Set up the mask
            self.mask = np.zeros_like(current_frame)

            # Skip the rest of the function, no need to do anything with it.
            return

        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, current_gray, self.next_feats, None, **self.lk_params)

        if p1 is not None:
            good_new = p1[st == 1]
            good_old = self.next_feats[st == 1]

        self.next_feats = cv2.goodFeaturesToTrack(current_gray, mask=None, **self.feature_params)

        # Keep track of old gray frame
        self.old_gray = current_gray.copy()

        if p1 is None:
            self.displacements = None
            self.p0 = None
            return

        calib_new = self.calibrate_pixels(good_new)
        calib_old = self.calibrate_pixels(good_old)

        self.displacements = calib_new - calib_old

        # Only keep the good features for the next iteration
        self.p0 = good_new.reshape(-1, 1, 2)
        self.p0_calib = calib_new.reshape(-1, 1, 2)

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            self.mask = cv2.line(self.mask, (int(a), int(b)), (int(c), int(d)), self.colors[0].tolist(), 1)
            current_frame = cv2.circle(current_frame, (int(a), int(b)), 2, self.colors[1].tolist(), -1)
        if self.SHOW_TRACKS:
            current_frame = cv2.add(current_frame, self.mask)

        # Display image
        if self.SHOW_IMAGE:
            cv2.imshow("camera", current_frame)
            cv2.waitKey(1)

    # Function called to publish the next frame of the recording
    def publish_data(self):
        #dimensions of the image for the algorithm
        dim=(640,480)
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

        self.prev_num = self.line_num

        return resized_img
    
    # Function to summarize the features and their velocities and add them to the file
    def process_features(self) -> None:
        if self.displacements is None or len(self.p0) < 4:
            self.line_num += 1
            return
        with open(self.dirToRead + "naive_calib_feat_data/" + self.featfilename + ".csv", "w", newline="") as featDataFile:
            dataWriter = csv.writer(featDataFile)
            header = ["pos_x", "pos_y", "vel_x", "vel_y"]
            dataWriter.writerow(header)
            del_t = int(self.record_lines[self.line_num]["TimeStamp"]) - int(self.record_lines[self.line_num-1]["TimeStamp"])
            del_t /= 1000

            for i, feat in enumerate(self.p0_calib):
                feat=np.reshape(feat, (2))
                row = [feat[0], feat[1], self.displacements[i][0]/del_t, self.displacements[i][1]/del_t]
                dataWriter.writerow(row)
        self.line_num += 1

    def process_images(self):
        while self.line_num < len(self.record_lines):
            img = self.publish_data()
            self.track_features(img)
            self.process_features()


if __name__=="__main__":
    tracker = NaiveFeatureTracker()
    tracker.process_images()