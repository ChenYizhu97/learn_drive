#!/usr/bin/env python3
import rospy
import torch
import numpy as np
from PIL import Image
from math import pi
from train_model import model
from train_model import tsfm
from myT import Robot



if __name__ == "__main__":

    # load model state, no gpu needed during predict.
    model.load_state_dict(torch.load('model_state', map_location=torch.device('cpu')))
    model.eval()
    
    rospy.init_node('drive_by_model', anonymous=True)
    robot = Robot('thymio')
    

    while True:
        
        # read image from camera and do transform
        image = robot.read_camera()
        image = Image.fromarray(image)
        image = tsfm(image)
        image = image.unsqueeze(0)
        
        # perdict labels      
        predicts = model(image).detach().reshape(-1)

        T_predict = predicts[0]
        C_predict = predicts[1]

        # calculate real labels for comparing
        sensor_vector = robot.read_sensor()
        T_real = np.dot(sensor_vector, [1, 2, 0, -2, -1])/3
        C_real = np.dot(sensor_vector, [-1, -1, 4, -1, -1])/4  
        
        rospy.loginfo('The predict T is {:.4f}, the real T is {:.4f},\n The predict C is {:.4f}, The real C is {:.4f}'.format(T_predict, T_real, C_predict, C_real))
        
        # drive the robot using predict labels
        ang_vel =  T_predict * 6 * pi
        lin_vel = (1 - abs(C_predict))*0.2
    
        if C_predict < -0.15 or C_predict > 0.4:
            for i in range(0, 3):
                robot.send_velocity(0, 2*pi)
        
        robot.send_velocity(lin_vel, ang_vel)
