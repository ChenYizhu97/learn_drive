#!/usr/bin/env python3
from myT import Robot
from math import pi
import rospy
import numpy as np

rospy.init_node('drive_by_sensor', anonymous=True)
robot = Robot('thymio', rate=6)

while True: 
    sensor_vector = robot.read_sensor()

    rotate = False

    T = np.dot(sensor_vector, [1, 2, 0, -2, -1])/3
    C = np.dot(sensor_vector, [-1, -1, 4, -1, -1])/4
   
    # set velocity	
    ang_vel =  T * 6 * pi
    lin_vel = (1 - abs(C))*0.2
    rospy.loginfo('T is {}, C is {}, the angular velocity is {}, the linear velocity is {}, the sensor vector is {}'.format(T, C, ang_vel, lin_vel, sensor_vector))
    
    # rotate in some situations
    if C < -0.15 or C > 0.4:
        for i in range(0, 3):
            robot.send_velocity(0, 2*pi)


    robot.send_velocity(lin_vel, ang_vel)
