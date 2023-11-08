from pathlib import Path
import struct
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# This file creates a plot of the depth image given in the 
# "depth_filename" variable

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

depth_filename = "/home/james/Pictures/img_SimpleFlight_0_2_1677783850436472400.pfm"

pfm = read_pfm(depth_filename)

pfm = np.clip(pfm, 0, 100)

sns.heatmap(pfm, cmap=sns.color_palette("Spectral_r", as_cmap=True))
plt.yticks([],[])
plt.xticks([],[])
plt.show()