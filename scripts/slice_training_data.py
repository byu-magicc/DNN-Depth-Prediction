import csv
import random

directory = "/home/james/Documents/AirSim/supercomputer_recording/"
file_to_write = directory + "calib_nn_data_part4.csv"

with open(directory+"calib_nn_data.csv", "r") as big_data_file:
    writer = csv.writer(open(file_to_write, "w", newline=""))
    reader = csv.DictReader(big_data_file)
    start = 360000
    stop = 480000

    i = 0
    lines_written = 0
    headers = reader.fieldnames
    writer.writerow(headers)
    # rand = random.randint(0, 20)
    # lines = []
    for line in reader:
        i += 1
        if i >= stop:
            break
        if i > start:
            line_to_write=[]
            for item in headers:
                line_to_write.append(line[item])
            # lines.append(line_to_write)
            # rand=random.randint(0, 20)
            writer.writerow(line_to_write)
            lines_written += 1
    # random.shuffle(lines)
    print("Wrote " + str(lines_written) + " lines")
    # for line in lines:
    #     writer.writerow(line)