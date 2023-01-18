# %%

import numpy as np
import matplotlib.pyplot as plt
import csv

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

data_filename = directory + "soph_nn_data.csv"
data_file = open(data_filename,"r", newline='')
data_reader = csv.DictReader(data_file, quoting=csv.QUOTE_NONNUMERIC)

column_names = ["depth","PXx", "PXy","Velx", "Vely", "Velz", "Wx", "Wy", "Wz","F1x", "F1y", "F1vx", "F1vy", "F2x", "F2y", "F2vx", "F2vy", "F3x", "F3y", "F3vx", "F3vy", "F4x", "F4y", "F4vx", "F4vy"]

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

velx = None
vely = None
velz = None

for line in data_reader:
    if len(line.keys()) < 10:
        continue
    if (not velx == line["Velx"]) and (not vely == line["Vely"]) and (not velz == line["Velz"]):
        velx = line["Velx"]
        vely = line["Vely"]
        velz = line["Velz"]
        quat = np.array([line["Q_W"],
                         line["Q_X"],
                         line["Q_Y"],
                         line["Q_Z"]])
        rot_mat = qaut_to_rotation(quat)
        velocity_global = np.array([[velx],
                            [vely],
                            [velz]])
        velocity_body = rot_mat @ velocity_global
        velocity_camera = velocity_body#TODO: rotate velocity into camera frame?
    tcol_sum = 0
    for num in range(1,5):
        featx = line["F" + str(num) + "x"]
        featy = line["F" + str(num) + "y"]
        featvx = line["F" + str(num) + "vx"]
        featvy = line["F" + str(num) + "vy"]
        feat_position = np.array([[featx], [featy]])
        feat_velocity = np.array([[featvx], [featvy]])

        calibrated_feat_pos = calibrate_pixels(feat_position)
        calibrated_feat_vel = calibrate_pixels(feat_velocity)

        tcolx = calibrated_feat_pos[0]/calibrated_feat_vel[0]
        tcoly = calibrated_feat_pos[1]/calibrated_feat_vel[1]

        tcol_sum += (tcolx + tcoly)/2
    tcol = tcol_sum/4

    pixel_pos = np.array([[line["PXx"]],[line["PXy"]],[1]])
    calibrated_pixel_pos = cam_inv @ pixel_pos


    vel_comp = calibrated_pixel_pos.T @ velocity_camera
    depth = vel_comp * tcol
