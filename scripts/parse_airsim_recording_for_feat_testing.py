import csv
import numpy as np
import cv2
from pathlib import Path
import struct
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from kd_tree_class import KD_Tree

def read_pfm(filename):
    with Path(filename).open('rb') as pfm_file:

        line1, line2, line3 = (pfm_file.readline().decode('latin-1').strip() for _ in range(3))
        assert line1 in ('PF', 'Pf')
        
        channels = 3 if "PF" in line1 else 1
        width, height = (int(s) for s in line2.split())
        scale_endianess = float(line3)
        bigendian = scale_endianess > 0
        scale = abs(scale_endianess)

        buffer = pfm_file.read()
        samples = width * height * channels
        assert len(buffer) == samples * 4
        
        fmt = f'{"<>"[bigendian]}{samples}f'
        decoded = struct.unpack(fmt, buffer)
        shape = (height, width, 3) if channels == 3 else (height, width)
        return np.reshape(decoded, shape) * scale

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
def uncalibrate_pixels(pos):
    new_pos = np.reshape(pos, (2, -1))
    uncalibrated = np.ones((3,1))
    uncalibrated[0:2] = new_pos

    calibrated = camera_intrinsics @ uncalibrated
    return np.reshape(calibrated[0:2], pos.shape)

directory = "/home/james/Documents/AirSim/supercomputer_recording/"
writing_filename = "testing_feat_depth.csv"
feat_directory = "feat_data/"

header = ["\"depth\"","\"PXx\"", "\"PXy\"", "\"Q_W\"", "\"Q_X\"", "\"Q_Y\"", "\"Q_Z\"", "\"Velx\"", "\"Vely\"", "\"Velz\"", "\"Wx\"", "\"Wy\"", "\"Wz\"","\"F1x\"", "\"F1y\"", "\"F1vx\"", "\"F1vy\""]

record_lines = []

try:
    recording_file = open(directory + "airsim_rec.txt","r")
    data_write_file = open(directory+writing_filename, "w", newline="")
    dataWriter = csv.writer(data_write_file)
    data_reader = csv.DictReader(recording_file, delimiter="\t")
    for line in data_reader:
        record_lines.append(line)
    recording_file.close()

    line_num = 0
    dataWriter.writerow(header)

    for line in record_lines:
        line_num += 1
        both_images = line["ImageFile"]

        # get the color image filename
        depth_image_filename = both_images.split(";")[1]

        # save the base of the filename for later
        featfilename = both_images.split(";")[0].split(".")[0] + ".csv"

        try:
            feat_file = open(directory+ feat_directory + featfilename)
            feat_reader = csv.DictReader(feat_file)
            pixels = read_pfm(directory + "images/" + depth_image_filename)
            # normalized = cv2.normalize(np.clip(pixels, 0, 100),None,255,0,cv2.NORM_MINMAX,cv2.CV_8UC1)
            # normalized = cv2.cvtColor(normalized,cv2.COLOR_GRAY2BGR)
            
            for feat in feat_reader:
                featx = float(feat["pos_x"])
                featy = float(feat["pos_y"])
                feat_pos = np.array([featx, featy])
                uncal_point = uncalibrate_pixels(feat_pos)
                x = int(uncal_point[0]/2)
                x = np.clip(x,0,319)
                y = int(uncal_point[1]/2)
                y = np.clip(y,0,239)

                # new_depth=cv2.circle(normalized.copy(),(x,y),2,(0,0,255),2)
                # cv2.imshow("window", new_depth)
                # cv2.waitKey(1000)
                
                depth = pixels[y,x] # does this for sure get the right depth?
                row = [depth, featx, featy, line["Q_W"], line["Q_X"], line["Q_Y"], line["Q_Z"], line["Velx"], line["Vely"], line["Velz"], line["Wx"], line["Wy"], line["Wz"]]
                row.append(feat["pos_x"])
                row.append(feat["pos_y"])
                row.append(feat["vel_x"])
                row.append(feat["vel_y"])
                dataWriter.writerow(row)
            feat_file.close()
            print("Finished line " + str(line_num))
            
        except FileNotFoundError:
            # no feature velocity data for this frame, skip it
            continue

except FileNotFoundError:
    print("Files not found")