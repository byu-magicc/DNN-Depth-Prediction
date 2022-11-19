import csv
import random

directory = "/home/james/Documents/AirSim/supercomputer_recording/"
file_to_write = directory + "nn_data_part1.csv"

with open(directory+"nn_data.csv", "r") as big_data_file:
    writer = csv.writer(open(file_to_write, "w", newline=""))
    reader = csv.DictReader(big_data_file)
    stop = 100000

    i = 0
    lines_written = 0
    headers = reader.fieldnames
    writer.writerow(headers)
    rand = random.randint(0, 20)
    for line in reader:
        i += 1
        if lines_written >= stop:
            break
        if (i+rand) % 20 == 0:
            line_to_write=[]
            for item in headers:
                line_to_write.append(line[item])
            writer.writerow(line_to_write)
            rand=random.randint(0, 20)
            lines_written += 1
