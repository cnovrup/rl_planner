import rospy
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDrive
import tf
import numpy as np
import math
from std_srvs.srv import Empty
from std_msgs.msg import Float32, Bool
from nav_msgs.msg import Path
from gazebo_msgs.srv import GetModelState, SpawnModel, SetModelState
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import JointState, LaserScan
import rospkg
from geometry_msgs.msg import Pose, PoseStamped
import tf.transformations
import random
from rosgraph_msgs.msg import Clock


class GazeboInterface:
    def __init__(self) -> None:
        self.ready = False
        self.odom_msg = Odometry()
        rospy.init_node('interface',log_level=rospy.INFO)
        self.odom_sub = rospy.Subscriber("/fendt_942_ad/odom", Odometry, callback=self.set_odom)
        self.joint_states_sub = rospy.Subscriber("/fendt_942_ad/joint_states", JointState, callback=self.set_wheel_angle)
        self.laser_sub = rospy.Subscriber("/scan", LaserScan, callback=self.set_laser)
        self.clock_sub = rospy.Subscriber("/clock", Clock, callback=self.set_time)
        self.acker_pub = rospy.Publisher("/fendt_942_ad/cmd_vel", AckermannDrive, queue_size=1)
        self.reward_pub = rospy.Publisher("/reward", Float32, queue_size=1)
        self.path_pub = rospy.Publisher("/target_path", Path, queue_size=1)
        self.reset_pub = rospy.Publisher("/reset_sim", Bool, queue_size=1)
        self.path_completed_pub = rospy.Publisher("/path_completed", Bool, queue_size=10)

        rospy.wait_for_service("/gazebo/reset_simulation")
        rospy.wait_for_service("/gazebo/unpause_physics")
        self.reset_gazebo = rospy.ServiceProxy("/gazebo/reset_simulation", Empty)
        self.unpause_gazebo = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause_gazebo = rospy.ServiceProxy("/gazebo/pause_physics", Empty)

        rospy.wait_for_service('/gazebo/get_model_state')
        self.state_srv = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)

        self.set_state_srv = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        rospack = rospkg.RosPack()

        self.tractor_x = 0
        self.tractor_y = 0
        self.tractor_th = 0
        self.tractor_v = 0
        self.tractor_d = 0
        self.prev_goal_tractor = None
        self.goal_tractor = None

        self.time = 0
        
        self.ready = True
        self.laser_ranges_normalized = [1]*512 

    def publish_path(self, path):
        path_msg = Path()
        path_msg.header.frame_id = "map"
        pose = PoseStamped()
        pose.pose.position.x = 0
        pose.pose.position.y = 0
        pose.pose.orientation.w = 1
        pose.header.frame_id = "map"
        path_msg.poses.append(pose)

        for wp in path:
            pose = PoseStamped()
            pose.pose.position.x = wp[0]
            pose.pose.position.y = wp[1]
            pose.pose.orientation.w = 1
            pose.header.frame_id = "map"
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)

    def path_completed(self):
        msg = Bool()
        msg.data = True
        self.path_completed_pub.publish(msg)

    def set_time(self, msg:Clock):
        self.time = msg.clock.secs + msg.clock.nsecs*1e-9

    def set_wheel_angle(self,msg:JointState):
        d1 = msg.position[0]
        d2 = msg.position[1]
        self.tractor_d = (d1 + d2)/2
     
    def set_laser(self, msg:LaserScan):
        self.laser_ranges = msg.ranges
        self.laser_ranges_normalized = []
        for range in msg.ranges:
            if(range == float('inf') or range > 30):
                self.laser_ranges_normalized.append(1)
            else:
                self.laser_ranges_normalized.append(range/30)

    def set_odom(self,msg:Odometry):
        self.odom_msg = msg
        self.tractor_x = msg.pose.pose.position.x
        self.tractor_y = msg.pose.pose.position.y
        self.tractor_v = msg.twist.twist.linear.x

        quaternion = (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        self.tractor_th = euler[2]
        
    def set_action(self, speed, angle):
        acker_msg = AckermannDrive()
        acker_msg.steering_angle = angle
        acker_msg.speed = speed
        self.acker_pub.publish(acker_msg)

    def play_sim(self):
        rospy.wait_for_service("/gazebo/unpause_physics")
        self.unpause_gazebo()

    def pause_sim(self):
        rospy.wait_for_service("/gazebo/pause_physics")
        self.pause_gazebo()

    def reset_sim(self):
        rospy.wait_for_service("/gazebo/reset_world")
        self.reset_gazebo()
        #self.pause_gazebo()

        reset_msg = Bool()
        reset_msg.data = True
        self.reset_pub.publish(reset_msg)     