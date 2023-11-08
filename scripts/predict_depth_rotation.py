# %%

import numpy as np
import matplotlib.pyplot as plt
import csv
from bin_for_heatmap import *
import seaborn as sns
from matplotlib.colors import LogNorm

# This file takes the data from parse_airsim_recording_for_feat_testing.py
# and predicts the depth using traditional equations. It then creates
# a heatmap for the predicted vs true depth.

def qaut_to_rotation(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

directory = "/home/james/Documents/AirSim/supercomputer_recording/"

data_filename = directory + "testing_feat_depth.csv"
data_file = open(data_filename,"r", newline='')
data_reader = csv.DictReader(data_file, quoting=csv.QUOTE_NONNUMERIC)

# column_names = ["depth","PXx", "PXy","Velx", "Vely", "Velz", "Wx", "Wy", "Wz","F1x", "F1y", "F1vx", "F1vy", "F2x", "F2y", "F2vx", "F2vy", "F3x", "F3y", "F3vx", "F3vy", "F4x", "F4y", "F4vx", "F4vy"]

f = 465.6
camera_intrinsics = np.array([[f, 0, 320],
                              [0, f, 240],
                              [0, 0, 1]])

cam_inv = np.linalg.inv(camera_intrinsics)

def calibrate_pixels(pos):
    uncalibrated = np.ones((3,1))
    uncalibrated[0:2] = pos

    calibrated = cam_inv @ uncalibrated
    return calibrated[0:2, 0]

g_velx = None
g_vely = None
g_velz = None

predicted_depth = []
actual_depth = []
body_to_camera = np.array([[0, 1, 0],
                           [0, 0, 1],
                           [1, 0, 0]])
pxc = 0.3
pyc = 0.
pzc = 0.


for line in data_reader:
    if len(line.keys()) < 10:
        continue
    if (not g_velx == line["\"Velx\""]) and (not g_vely == line["\"Vely\""]) and (not g_velz == line["\"Velz\""]):
        # updated = True
        g_velx = line["\"Velx\""]
        g_vely = line["\"Vely\""]
        g_velz = line["\"Velz\""]
        quat = np.array([line["\"Q_W\""],
                         line["\"Q_X\""],
                         line["\"Q_Y\""],
                         line["\"Q_Z\""]])
        rot_mat = qaut_to_rotation(quat)
        velocity_global = np.array([[g_velx],
                            [g_vely],
                            [g_velz]])
        velocity_body = rot_mat.T @ velocity_global
        velocity_camera = np.reshape(body_to_camera @ velocity_body,(3))
        velx = velocity_camera[0]
        vely = velocity_camera[1]
        velz = velocity_camera[2]
        omega_body = np.array([[line["\"Wx\""]],[line["\"Wy\""]],[line["\"Wz\""]]])
        omega_camera = np.reshape(body_to_camera @ omega_body,(3))
        wx = omega_camera[0]
        wy = omega_camera[1]
        wz = omega_camera[2]
    # depth_sum = 0
    # if not updated or np.linalg.norm(omega_camera) > 0.0173:
    #     updated = False
    #     continue
    if np.sqrt((line["\"F1x\""] - line["\"PXx\""])**2 + (line["\"F1y\""] - line["\"PXy\""])**2) > 0.00283:
        continue
    # for num in range(1,5):
    num = 1
    featx = line["\"F" + str(num) + "x\""]
    featy = line["\"F" + str(num) + "y\""]
    featvx = line["\"F" + str(num) + "vx\""]
    featvy = line["\"F" + str(num) + "vy\""]
    feat_position = np.array([featx, featy])
    feat_velocity = np.array([featvx, featvy])

    # feat_position = calibrate_pixels(feat_position)
    # feat_velocity = calibrate_pixels(feat_velocity)

    pz_x = (pyc*wz - pzc*wy - velx-pxc*featx*wy + pyc*featx*wx+velz*featx)/(featvx - featy*wz + (1+featx**2)*wy - featx*featy*wx)
    pz_y =(-pxc*wz + pzc*wx - vely-pxc*featy*wy + pyc*featy*wx+velz*featy)/(featvy + featx*wz - (1+featy**2)*wx + featx*featy*wy)

    pz = (pz_x + pz_y)/2
    depth = pz*np.sqrt(1+featx**2+featy**2)
    # depth = depth_sum#/4

    # pixel_pos = np.array([[line["\"PXx\""]],[line["\"PXy\""]],[1]])
    # calibrated_pixel_pos = cam_inv @ pixel_pos

    # depth = (depth if depth <= 100 else 100) if depth >= 0 else 0
    depth = depth if depth <= 100 else 100
    predicted_depth.append(depth)
    depth = line["\"depth\""]
    depth = depth if depth <= 100 else 100
    actual_depth.append(depth)
#%%
a = plt.axes(aspect='equal')
plt.scatter(predicted_depth, actual_depth, 0.1)
plt.xlabel('True Values [m]')
plt.ylabel('Predictions [m]')
lims = [0, 100]
plt.xlim(lims)
plt.ylim(lims)
# error = 5
_ = plt.plot(lims, lims, "k")
# _ = plt.plot(lims, [lims[0]+error, lims[1] + error], "b")
# _ = plt.plot(lims, [lims[0]-error, lims[1]-error], "r")
#%%
relative_errors = np.abs((np.clip(np.array(predicted_depth),0,100) - np.array(actual_depth))/np.array(actual_depth))
avg_rel_error = np.sum(relative_errors)/len(predicted_depth)
print("The average relative error is " + str(avg_rel_error*100) + "%")
# %%
bins = bin_for_heatmap(actual_depth, predicted_depth)
ax = sns.heatmap(bins+0.1, norm=LogNorm())
ax.invert_yaxis()
plt.show()

# %%
