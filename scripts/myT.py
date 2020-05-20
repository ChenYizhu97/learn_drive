import rospy
import numpy as np
from rospy.numpy_msg import numpy_msg
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Range
from sensor_msgs.msg import Image
from math import pi

class Robot(object):
    
    def __init__(self, name, rate=12):
        
        self.name = name
        self.camera_img = Image()
        self.sensor1 = 0.12
        self.sensor2 = 0.12
        self.sensor3 = 0.12
        self.sensor4 = 0.12
        self.sensor5 = 0.12
        self.rate = rospy.Rate(rate)

        self.camera_subscriber = rospy.Subscriber(name+'/camera/image_raw', numpy_msg(Image), self._update_camera)
        self.sensor1_subscriber = rospy.Subscriber(name+'/proximity/left', Range, self._update_sensor1)   
        self.sensor2_subscriber = rospy.Subscriber(name+'/proximity/center_left', Range, self._update_sensor2)
        self.sensor3_subscriber = rospy.Subscriber(name+'/proximity/center', Range, self._update_sensor3)
        self.sensor4_subscriber = rospy.Subscriber(name+'/proximity/center_right', Range, self._update_sensor4)
        self.sensor5_subscriber = rospy.Subscriber(name+'/proximity/right', Range, self._update_sensor5) 

        self.velocity_publisher = rospy.Publisher(name+'/cmd_vel', Twist, queue_size=10)
       

        super().__init__()

    
    def read_camera(self, ):
        img = np.frombuffer(self.camera_img.data, dtype=np.uint8).reshape(self.camera_img.height, self.camera_img.width, -1)
        return img
    
    def read_sensor(self, ):
        sensor_vector = [self.sensor1/0.12, self.sensor2/0.12, self.sensor3/0.12, self.sensor4/0.12, self.sensor5/0.12,]
        return np.asarray(sensor_vector)

    def send_velocity(self, lin_vel, ang_vel, ):
        vel_msg = Twist()
        vel_msg.linear.x = lin_vel
        vel_msg.angular.z = ang_vel
        
        self.velocity_publisher.publish(vel_msg)
        self.rate.sleep()
               
    def _update_camera(self, data, ):
        self.camera_img = data

    def _update_sensor1(self, data, ):
        self.sensor1 = data.range

    def _update_sensor2(self, data,):
        self.sensor2 = data.range
    
    def _update_sensor3(self, data,):
        self.sensor3 = data.range

    def _update_sensor4(self, data, ):
        self.sensor4 = data.range

    def _update_sensor5(self, data, ):
        self.sensor5 = data.range
