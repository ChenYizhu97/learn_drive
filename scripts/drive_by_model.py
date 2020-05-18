import rospy
import torch
import numpy as np
from PIL import Image
from math import pi
from train_model import model
from train_model import tsfm
from myT import Robot



if __name__ == "__main__":
    model.load_state_dict(torch.load('model_state'))
    rospy.init_node('drive_by_model', anonymous=True)
    robot = Robot('thymio')
    model.eval()

    while True:
        image = robot.read_camera()
        image = Image.fromarray(image)
        image = tsfm(image)
        image = image.unsqueeze(0)       
        T_predict = model(image).detach().numpy().item()
        T_real = np.dot(robot.read_sensor(),[1, 2, 0, -2, -1])
        
        rospy.loginfo('The predict T is {:.4f}, the real T is {:.4f}'.format(T_predict, T_real))

        ang_vel = -T_predict * pi
        lin_vel = 0.2
        
        robot.send_velocity(lin_vel, ang_vel)
