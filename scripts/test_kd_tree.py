import csv
import numpy as np
from pathlib import Path
import struct
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from kd_tree_class import KD_Tree

directory = "/home/james/Documents/AirSim/supercomputer_recording/"
feat_directory = "feat_data/"

f = 465.6
camera_intrinsics = np.array([[f, 0, 320],
                              [0, f, 240],
                              [0, 0, 1]])

cam_inv = np.linalg.inv(camera_intrinsics)

def calibrate_pixels(pos):
    new_pos = np.reshape(pos, (2, -1))
    uncalibrated = np.ones((3,1))
    uncalibrated[0:2] = new_pos

    calibrated = cam_inv @ uncalibrated
    return np.reshape(calibrated[0:2], pos.shape)

record_lines = []

try:
    recording_file = open(directory + "airsim_rec.txt","r")
    data_reader = csv.DictReader(recording_file, delimiter="\t")
    for line in data_reader:
        record_lines.append(line)
    recording_file.close()

    line_num = 0

    for line in record_lines:
        line_num += 1
        both_images = line["ImageFile"]

        # get the color image filename
        depth_image_filename = both_images.split(";")[1]

        color_filename = both_images.split(";")[0]

        # save the base of the filename for later
        featfilename = both_images.split(";")[0].split(".")[0] + ".csv"

        try:
            feat_file = open(directory+ feat_directory + featfilename)
            feat_reader = csv.DictReader(feat_file)
            feats = []
            for feat in feat_reader:
                feat["pos_x"] = float(feat["pos_x"])
                feat["pos_y"] = float(feat["pos_y"])
                feats.append(feat)
            feat_file.close()

            tree = KD_Tree(feats, ["pos_x", "pos_y"])

            for i in range(0, 639, 2):
                kd_closest = []
                for j in range(0, 479, 2):
                    uncal_point = np.array([2*i+1,2*j+1])
                    cal_point = calibrate_pixels(uncal_point)
                    x = cal_point[0]
                    y = cal_point[1]
                    brute_force_closest=[]
                    for brute_feat in feats:
                        dist = np.sqrt((x - brute_feat["pos_x"])**2 + (y - brute_feat["pos_y"])**2)
                        brute_feat["dist"] = dist
                        brute_force_closest.append(brute_feat)
                    brute_force_closest = sorted(brute_force_closest, key=lambda x:x["dist"])

                    point = {"pos_x":x, "pos_y":y}
                    kd_closest = tree.find_nearest_neighbors(point, kd_closest)

                    if len(brute_feat) < 4:
                        print("Brute force failed to produce 4 features")
                    if len(kd_closest) < 4:
                        print("KD-Tree failed to produce 4 features on line " + str(line_num))
                    for brute_feat, kd_feat in zip(brute_force_closest[0:4],kd_closest):
                        if (not brute_feat["pos_x"] == kd_feat["pos_x"] or
                        not brute_feat["pos_y"] == kd_feat["pos_y"] or
                        not brute_feat["vel_x"] == kd_feat["vel_x"] or
                        not brute_feat["vel_y"] == kd_feat["vel_y"]):
                            print("Mistake made on line " + str(line_num))
                            break
            print("Finished line " + str(line_num))
            
            
        except FileNotFoundError:
            # no feature velocity data for this frame, skip it
            continue

except FileNotFoundError:
    print("Files not found")