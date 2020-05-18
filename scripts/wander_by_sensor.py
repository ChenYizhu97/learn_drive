from myT import Robot
import math
import rospy

rospy.init_node('wander_by_sensor', anonymous=True)
robot = Robot('thymio', rate=12)

lin_vel = 0.2
ang_vel = math.pi        
        
while True: 
    sensor_vector = robot.read_sensor()
    rotate = False
    for sensor in sensor_vector:
        if sensor < 0.1:
            rotate = True
    
    if rotate:
        robot.send_velocity(0, ang_vel)


    else:
        robot.send_velocity(lin_vel, 0)
 