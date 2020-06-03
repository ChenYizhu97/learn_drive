#!/usr/bin/env python3
from myT import Robot
import os
import re
import rospy
import numpy as np
import time
import random
from PIL import Image


rospy.init_node('collect_data', anonymous=True)

robot = Robot('thymio', rate=12)

# path of data set
sample_number = 0
images_dir = 'dataset/images/'
labels_dir = 'dataset/labels/'

T_labels = []
C_labels = []
T_vector = np.asarray([1, 2, 0, -2, -1])
C_vector = np.asarray([-1, -1, 4, -1, -1])

# get the based image index
image_names = os.listdir(images_dir)
if len(image_names) == 0:
    base_idx = 0
else:
    image_id = [ int(re.findall(r'\d+', image_name)[0]) for image_name in image_names]
    base_idx = max(image_id) + 1

# collect samples
while sample_number < 1000:
    image_np = robot.read_camera()
    sensor_vector = robot.read_sensor()
    
    image = Image.fromarray(image_np)
    img_path = '{}image{}.png'.format(images_dir, base_idx+sample_number)
    
    T_label = np.dot(sensor_vector, T_vector)/3
    C_label = np.dot(sensor_vector, C_vector)/4

    # drop samples that have no obstacles with a probability of 0.9.
    if abs(C_label) < 0.001:
        if random.randint(1, 100) < 100:
            time.sleep(0.1)
            continue

    T_labels.append(T_label)
    C_labels.append(C_label)
    # save image
    image.save(img_path)
    sample_number = sample_number + 1
    print('collected the {} sample...'.format(sample_number))
    time.sleep(0.1)

# save labels
with open(labels_dir+'T_labels.csv', 'ba') as T_labels_file:
    np.savetxt(T_labels_file, T_labels, delimiter='\n')


with open(labels_dir+'C_labels.csv', 'ba') as C_labels_file:
    np.savetxt(C_labels_file, C_labels, delimiter='\n')
