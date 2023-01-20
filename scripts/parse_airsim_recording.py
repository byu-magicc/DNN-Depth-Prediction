import csv
import numpy as np
from pathlib import Path
import struct
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
writing_filename = "calib_nn_data.csv"
feat_directory = "naive_calib_feat_data/"

header = ["depth","PXx", "PXy", "Q_W", "Q_X", "Q_Y", "Q_Z", "Velx", "Vely", "Velz", "Wx", "Wy", "Wz","F1x", "F1y", "F1vx", "F1vy", "F2x", "F2y", "F2vx", "F2vy", "F3x", "F3y", "F3vx", "F3vy", "F4x", "F4y", "F4vx", "F4vy"]

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

        color_filename = both_images.split(";")[0]

        # save the base of the filename for later
        featfilename = both_images.split(";")[0].split(".")[0] + ".csv"

        try:
            feat_file = open(directory+ feat_directory + featfilename)
            feat_reader = csv.DictReader(feat_file)
            feats = []
            for feat in feat_reader:
                feats.append(feat)
            feat_file.close()
            pixels = read_pfm(directory + "images/" + depth_image_filename)

            # plt.imshow(mpimg.imread(directory+"images/"+color_filename))
            # plt.show()
            # plt.imshow(pixels)
            # plt.show()


            for i in range(0, pixels.shape[0]-1, 2):
                # closest_feats = []
                # for feat in feats:
                #     if len(closest_feats) < 4:
                #         dist = np.sqrt((i - feat["pos_x"])**2 + feat["pos_y"]**2)
                #         feat["dist"] = dist
                #         closest_feats.append(feat)
                #         continue
                #     dist = np.sqrt((i - feat["pos_x"])**2 + feat["pos_y"]**2)
                #     i = 4
                #     while dist < closest_feats[i-1]["dist"]:
                #         i -= 1
                #     if i != 4:
                #         feat["dist"] = dist
                #         closest_feats.insert(i, feat)
                #         closest_feats.pop()
                for j in range(0, pixels.shape[1]-1, 2):
                    x = 2*i+1
                    y = 2*j+1
                    closest_feats = []
                    for feat in feats:
                        if len(closest_feats) < 4:
                            dist = np.sqrt((x - float(feat["pos_x"]))**2 + (y -float(feat["pos_y"]))**2)
                            feat["dist"] = dist
                            closest_feats.append(feat)
                            continue
                        dist = np.sqrt((x - float(feat["pos_x"]))**2 + (y - float(feat["pos_y"]))**2)
                        pos = 4
                        while dist < closest_feats[pos-1]["dist"] and pos > 0:
                            pos -= 1
                        if pos != 4:
                            feat["dist"] = dist
                            closest_feats.insert(pos, feat)
                            closest_feats.pop()
                            
                    depth = np.min(pixels[i:i+2, j:j+2])
                    pos = np.array([x, y])
                    pos_calib = calibrate_pixels(pos)
                    row = [depth, pos_calib[0], pos_calib[1], line["Q_W"], line["Q_X"], line["Q_Y"], line["Q_Z"], line["Velx"], line["Vely"], line["Velz"], line["Wx"], line["Wy"], line["Wz"]]
                    for feat in closest_feats:
                        row.append(feat["pos_x"])
                        row.append(feat["pos_y"])
                        row.append(feat["vel_x"])
                        row.append(feat["vel_y"])
                    dataWriter.writerow(row)
            print("Finished line " + str(line_num))
            
        except FileNotFoundError:
            # no feature velocity data for this frame, skip it
            continue

except FileNotFoundError:
    print("Files not found")