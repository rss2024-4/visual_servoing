#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np
from numpy.linalg import norm
from math import copysign

from vs_msgs.msg import ConeLocation, ParkingError
from ackermann_msgs.msg import AckermannDriveStamped

class ParkingController(Node):
    """
    A controller for parking in front of a cone.
    Listens for a relative cone location and publishes control commands.
    Can be used in the simulator and on the real robot.
    """
    
    STOP_EPS = 0 # prevent stopping
    
    def __init__(self):
        super().__init__("parking_controller")

        self.declare_parameter("drive_topic")
        DRIVE_TOPIC = self.get_parameter("drive_topic").value # set in launch file; different for simulator vs racecar

        self.drive_pub = self.create_publisher(AckermannDriveStamped, DRIVE_TOPIC, 10)
        self.create_subscription(ConeLocation, "/point", self.relative_cone_callback, 1)
        self.create_timer(0.05, self.timer_cb)
        self.fwd = np.array([1,0])
        self.L = .325
        self.speed = 4.0
        self.angle = 0.0
        self.get_logger().info("Point Follower Initialized")

    def relative_cone_callback(self, msg):
        target = np.array([msg.x_pos, msg.y_pos])
        # steer
        d = np.linalg.norm(target)
        sin_eta = np.sign(target[1])*norm(np.cross(self.fwd, target)) / d
        self.angle = np.math.atan2(2*self.L*sin_eta, d)

    def timer_cb(self):
        # make signal
        drive_cmd = AckermannDriveStamped()
        drive_cmd.drive.steering_angle = self.angle
        drive_cmd.drive.speed = self.speed
        self.drive_pub.publish(drive_cmd)

def main(args=None):
    rclpy.init(args=args)
    pc = ParkingController()
    rclpy.spin(pc)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
