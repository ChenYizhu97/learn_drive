from myT import Robot
import rospy
import numpy as np
import time
from PIL import Image


rospy.init_node('collect_data', anonymous=True)

robot = Robot('thymio', rate=12)

sample_number = 0
images_dir = 'images/'
labels_dir = 'labels/'

T_labels = []
C_labels = []
T_vector = np.asarray([1, 2, 0, -2, -1])
C_vector = np.asarray([-1, -1, 4, -1, -1])


while sample_number < 1000:
    image_np = robot.read_camera()
    sensor_vector = robot.read_sensor()
    
    image = Image.fromarray(image_np)
    img_path = '{}image{}.png'.format(images_dir, sample_number)
    image.save(img_path)

    T_label = np.dot(sensor_vector, T_vector)
    C_label = np.dot(sensor_vector, C_vector)
    T_labels.append(T_label)
    C_labels.append(C_label)
    
    sample_number = sample_number + 1
    print('collected the {} sample...'.format(sample_number))
    time.sleep(1)

np.savetxt(labels_dir+'T_labels.csv', T_labels, delimiter='\n')
np.savetxt(labels_dir+'C_labels.csv', C_labels, delimiter='\n')