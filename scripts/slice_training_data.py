import csv

directory = "/home/james/Documents/AirSim/supercomputer_recording/"
file_to_write = directory + "nn_data_part1.csv"

with open(directory+"nn_data.csv", "r", newline="") as big_data_file:
    writer = csv.writer(file_to_write)
    reader = csv.DictReader(big_data_file)
    start = 0
    stop = 10000

    i = 0
    headers = reader.fieldnames
    writer.writerow(headers)
    for line in reader:
        if i > stop:
            break
        if i < start:
            continue
        writer.writerow(line)
