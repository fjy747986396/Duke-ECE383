import numpy as np
import matplotlib.pyplot as plt


DATAFILE =  open('DataFile.txt','r')
config_record = []
for each_line in DATAFILE:
	if "[" in each_line:
		if ">" in each_line:
			each_line = each_line[3:-2].split(",")
		else:
			print("illegal Config" + str(each_line))
			each_line = each_line[1:-2].split(",")
		config_record.append([float(each) for each in each_line])
config_record = np.array(config_record)

for joint_idx in range(config_record.shape[1]):
    plt.figure(figsize=(10,5))
    plt.title("joint idx: " + str(joint_idx))
    plt.plot(config_record[:,joint_idx])