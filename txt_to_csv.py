#code to tranform txt files to csv with proper joint name headings

import pandas as pd
import csv
import itertools

def transform(file):
    column = ['head_x', 'head_y', 'head_z', 'neck_x', 'neck_y','neck_z', 'spine_x',
    'spine_y','spine_z', 'hip_x','hip_y','hip_z','shoulderL_x', 'shoulderL_y', 'shoulderL_z', 'shoulderR_x',
    'shoulderR_y', 'shoulderR_z', 'elbowL_x', 'elbowL_y', 'elbowL_z', 'elbowR_x',
    'elbowR_y', 'elbowR_z','wristL_x','wristL_y','wristL_z','wristR_x','wristR_y',
    'wristR_z','handL_x', 'handL_y','handL_z','handR_x','handR_y','handR_z',
    'handtipL_x','handtipL_y', 'handtipL_z','handtipR_x','handtipR_y','handtipR_z'
    ,'hipL_x','hipL_y','hipL_z','hipR_x','hipR_y','hipR_z','kneeL_x','kneeL_y','kneeL_z'
    ,'kneeR_x','kneeR_y','kneeR_z','ankleL_x','ankleL_y','ankleL_z','ankleR_x','ankleR_y',
    'ankleR_z','footL_x','footL_y','footL_z','footR_x','footR_y','footR_z']


    #
    # with open(file, 'r') as in_file:
    with open(file) as fin, open('test.csv', 'w') as fout:
        o=csv.writer(fout)
        o.writerow(column)
        for line in fin:
            o.writerow(line.split())
        # lines = in_file.read().splitlines()
        # stripped = [line.replace(","," ").split() for line in lines]
        # grouped = itertools.izip(*[stripped]*1)
        # with open('test.csv', 'w') as out_file:
        #     writer = csv.writer(out_file)
        #     writer.writerow(column)
        #     for group in grouped:
        #         writer.writerows(group)

        # stripped = (line.strip() for line in in_file)
        # lines = (line.split(",") for line in stripped if line)
        # with open('test.csv', 'w') as out_file:
        #     writer = csv.writer(out_file)
        #     writer.writerow(column)
        #     writer.writerows(lines)
    # df = pd.read_csv(file, sep = " ", header = None)
    # df.to_csv('test.csv',  header = column)

transform('rex.txt')
