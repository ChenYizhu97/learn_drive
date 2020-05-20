from myT import Robot
import rospy
import numpy as np
import time
import random
from PIL import Image


rospy.init_node('collect_data', anonymous=True)

robot = Robot('thymio', rate=12)

sample_number = 0
images_dir = 'dataset/images/'
labels_dir = 'dataset/labels/'

T_labels = []
C_labels = []
T_vector = np.asarray([1, 2, 0, -2, -1])
C_vector = np.asarray([-1, -1, 4, -1, -1])


while sample_number < 1000:
    image_np = robot.read_camera()
    sensor_vector = robot.read_sensor()
    
    image = Image.fromarray(image_np)
    img_path = '{}image{}.png'.format(images_dir, sample_number)
    
    T_label = np.dot(sensor_vector, T_vector)/3
    C_label = np.dot(sensor_vector, C_vector)/4

    if abs(C_label) < 0.001:
        if random.randint(1, 100) < 100:
            time.sleep(1)
            continue

    T_labels.append(T_label)
    C_labels.append(C_label)
    image.save(img_path)
    sample_number = sample_number + 1
    print('collected the {} sample...'.format(sample_number))
    time.sleep(1)

np.savetxt(labels_dir+'T_labels.csv', T_labels, delimiter='\n')
np.savetxt(labels_dir+'C_labels.csv', C_labels, delimiter='\n')