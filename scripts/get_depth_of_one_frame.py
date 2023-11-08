#%%
import csv
import numpy as np
from pathlib import Path
import struct
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from kd_tree_class import KD_Tree
import time

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
from keras import layers
import pandas as pd
from bin_for_heatmap import *
import seaborn as sns
from matplotlib.colors import LogNorm

# This file takes a single frame from the AirSim recording (the line 
# given by target_line_number + 2, so if you want it to take line 
# 352, set target_line_number=350) and calculates the depth given 
# the features and velocities and the motion from that file line. 
# It then produces a plot of the actual depth and the predicted depth

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

directory = "/home/james/Documents/AirSim/supercomputer_recording/"
# writing_filename = "calib_nn_data.csv"
feat_directory = "feat_data/"

header = ["\"depth\"","\"PXx\"", "\"PXy\"", "\"Q_W\"", "\"Q_X\"", "\"Q_Y\"", "\"Q_Z\"", "\"Velx\"", "\"Vely\"", "\"Velz\"", "\"Wx\"", "\"Wy\"", "\"Wz\"","\"F1x\"", "\"F1y\"", "\"F1vx\"", "\"F1vy\"", "\"F2x\"", "\"F2y\"", "\"F2vx\"", "\"F2vy\"", "\"F3x\"", "\"F3y\"", "\"F3vx\"", "\"F3vy\"", "\"F4x\"", "\"F4y\"", "\"F4vx\"", "\"F4vy\""]
normalizer = tf.keras.layers.Normalization()
# normalizer.adapt(np.array(prediction_rows)) //example

def build_and_compile_model(norm):
    model = tf.keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                    optimizer=tf.keras.optimizers.Adam(0.001))
    return model

dnn_model = build_and_compile_model(normalizer)
# loads the weights from a checkpoint made during the training progress
dnn_model.load_weights("/home/james/catkin_ws/src/ttc-object-avoidance/scripts/training_calib_soph_f/cp_0029.ckpt")

record_lines = []

target_line_number = 224


recording_file = open(directory + "airsim_rec.txt","r")
# data_write_file = open(directory+writing_filename, "w", newline="")
# dataWriter = csv.writer(data_write_file)
data_reader = csv.DictReader(recording_file, delimiter="\t")
line_number = 0
for line in data_reader:
    # record_lines.append(line)
    if line_number == target_line_number:
        break
    line_number += 1
recording_file.close()

line_num = 0
# dataWriter.writerow(header)

    
print(line["ImageFile"])
both_images = line["ImageFile"]

# get the color image filename
depth_image_filename = both_images.split(";")[1]

color_filename = both_images.split(";")[0]

# save the base of the filename for later
featfilename = both_images.split(";")[0].split(".")[0] + ".csv"


feat_file = open(directory+ feat_directory + featfilename)
feat_reader = csv.DictReader(feat_file)
feats = []
for feat in feat_reader:
    feat["pos_x"] = float(feat["pos_x"])
    feat["pos_y"] = float(feat["pos_y"])
    feat["vel_x"] = float(feat["vel_x"])
    feat["vel_y"] = float(feat["vel_y"])
    feats.append(feat)
feat_file.close()
pixels = read_pfm(directory + "images/" + depth_image_filename)

st = time.process_time()

tree = KD_Tree(feats, ["pos_x", "pos_y"])
et = time.process_time()
print("KD-tree computation:", (et-st), "seconds")
prediction_rows = np.ones((19200,28))
# prediction_rows = []
prediction_rows[:,2] = float(line["Q_W"])
prediction_rows[:,3] = float(line["Q_X"])
prediction_rows[:,4] = float(line["Q_Y"])
prediction_rows[:,5] = float(line["Q_Z"])
prediction_rows[:,6] = float(line["Velx"])
prediction_rows[:,7] = float(line["Vely"])
prediction_rows[:,8] = float(line["Velz"])
prediction_rows[:,9] = float(line["Wx"])
prediction_rows[:,10] = float(line["Wy"])
prediction_rows[:,11] = float(line["Wz"])

for i in range(0, pixels.shape[0], 2): #the depth images are 320x240 while the images are 640x480
    closest_feats = []
    for j in range(0, pixels.shape[1], 2):
        uncal_point = np.array([2*i+1,2*j+1]) #uncalibrated pixel location
        cal_point = calibrate_pixels(uncal_point)
        x = cal_point[0]
        y = cal_point[1]
        point = {"pos_x":x, "pos_y":y}
        closest_feats = tree.find_nearest_neighbors(point, closest_feats)
        
        # depth = np.min(pixels[i:i+2, j:j+2])#TODO: Double check this is right (I think it should be y,x)
        # row = [x, y, line["Q_W"], line["Q_X"], line["Q_Y"], line["Q_Z"], line["Velx"], line["Vely"], line["Velz"], line["Wx"], line["Wy"], line["Wz"]]
        row = i//2*pixels.shape[1]//2+j//2
        prediction_rows[row,0] = x
        prediction_rows[row,1] = y
        co = 12
        for k,feat in enumerate(closest_feats):
            prediction_rows[row, co+4*k+0] = feat["pos_x"]
            prediction_rows[row, co+4*k+1] = feat["pos_y"]
            prediction_rows[row, co+4*k+2] = feat["vel_x"]
            prediction_rows[row, co+4*k+3] = feat["vel_y"]

#%%
prediction_st = time.process_time()
predicted_depths = dnn_model.predict(prediction_rows)
predicted_depths = np.reshape(predicted_depths, (pixels.shape[0]//2, pixels.shape[1]//2))
et = time.process_time()
print("Prediction processing time:", (et - prediction_st), "seconds")
print("Total processing time:", (et-st), "seconds")

#%%
pixels = np.clip(pixels, 0, 100)
sns.heatmap(pixels, cmap=sns.color_palette("Spectral_r", as_cmap=True))
plt.yticks([],[])
plt.xticks([],[])
plt.show()
    
#%%
sns.heatmap(predicted_depths, cmap=sns.color_palette("Spectral_r", as_cmap=True))
plt.yticks([],[])
plt.xticks([],[])
plt.show()
# %%
