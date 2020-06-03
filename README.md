# Intro
This the final project of 'Robotics-spring 2020' in Universita della Svizzera Italiana.
The main goal of this project is to trainning a CNN regressor which maps the each image captured by the robot into two values that are used to drive the robot avoiding obstacles.

# Enviroment
## Robot
The robot used is the Mighty Thymio (MyT). You need the catkin package https://github.com/jeguzzi/thymioid.git to run this project.
## World
To make this project feasible, some simplicity is made on the gazebo worlds used to train and test the model.
The worlds are closed areas which contains four kinds of simple obstacles. You can find the gazebo configuration files of these obstacles under the directory launch/models.
## Data set
10200 images and their corresponding labels. You can download the dataset on https://www.kaggle.com/dataset/7400c5212d2db9e2fb318cd22516181c2ced9bbfa11d135eeea72f875a688a9b .

For each image, the labels are calculated by the values returned by the frontal sensors of the robot. Let S denote the normalized vectors return by the sensors, which means the range value of each sensor is divided by its maximum range, then label T = dot(S, [1, 2, 0, -2, -1]) / 3 and label C = dot(S, [-1, -1, 4, -1, -1]) / 4.  
T-> 1 when obstacles are on the left.  T -> 0 when obstacles are symmetric w.r.t. sensors or there are no obstacles. T --> -1 when obstacles are on the right.
C -> 1 when obstacle is centered. C -> 0 when there is no obstacle. C --> -1 when obstacle is not centered.
The predicted labels are used to drive the robot avoiding obstacles.

# How to drive
- Linear velocity = (1-abs(C)) * C * v_0 
  Angular velocity = T * a_0
- Stop and rotate when C > threshold_0 or C < threshold_1 

# Model
The architecture of the model is ResNet18 -> BN(ReLU(Linear(1000, 512))) -> BN(ReLU(Linear(512, 128))) -> Tanh(Linear(512, 2)).
The ResNet18 extract features from the images and the three full connection layers do the regression. 
## train
The loss function is the mean L1 loss. The optimizer is the Adam optimizer with lr=0.0001. The batchsize during trainning is 64. After 100 epochs of trainning, the model seems could drive the robot avoiding obstacles for a certain time.
Here are the videos that shows the robot driving by model. https://youtu.be/X4u4RAbQHjw. https://youtu.be/oHs6NMXtruU. 

# Usage
- run 'roslaunch learn_drive thymio_gazebo_bringup.launch name:=thymio world:=simple' to start the simulation with training world. Change simple to triangle for running test world.
- run 'roslaunch learn_drive collect_dataset.launch' for collecting data set.
- run 'roslaunch learn_drive drive_by_model.launch' to drive robot by model. Make sure to run it under the path learn_drive/scripts/ , otherwise it cant find the model_state file.
- run 'roslaunch learn_drive drive_by_sensor.launch' to drive robot by sensor.

Don't change the name of robot, the scripts are not properly organised. If you change the name of robot, you also have to change that name in scripts.

Create the directories mentioned in the script 'collect_dataset.py' before you collect data. 
