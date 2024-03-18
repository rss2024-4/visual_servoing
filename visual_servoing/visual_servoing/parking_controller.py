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
    
    STOP_EPS = 0.1 # in meters
    
    def __init__(self):
        super().__init__("parking_controller")

        self.declare_parameter("drive_topic")
        DRIVE_TOPIC = self.get_parameter("drive_topic").value # set in launch file; different for simulator vs racecar

        self.drive_pub = self.create_publisher(AckermannDriveStamped, DRIVE_TOPIC, 10)
        self.error_pub = self.create_publisher(ParkingError, "/parking_error", 10)

        self.create_subscription(ConeLocation, "/relative_cone", self.relative_cone_callback, 1)

        self.parking_distance = .75 # meters; try playing with this number!
        self.relative_x = 0
        self.relative_y = 0
        self.fwd = np.array([1,0])
        self.L = .325
        self.cone_radius = 0.067437 # looked up some random cone online
        self.speed = 1.2

        self.get_logger().info("Parking Controller Initialized")

    def relative_cone_callback(self, msg):
        # get center of cone
        # cone_edge = np.array([msg.x_pos, msg.y_pos])
        # norm_cone = norm(cone_edge)
        # s_cone = np.sign(msg.y_pos)*norm(np.cross(self.fwd, cone_edge)) / norm_cone
        # c_cone = np.dot(self.fwd, cone_edge) / norm_cone
        # target = np.array([msg.x_pos + self.cone_radius*s_cone,             # this point should be the center of the cone
        #                    msg.y_pos + self.cone_radius*c_cone]).T
        target = np.array([msg.x_pos, msg.y_pos])

        # set for later
        self.relative_x, self.relative_y = target[0], target[1]
        
        # steer
        d = np.linalg.norm(target)
        sin_eta = np.sign(target[1])*norm(np.cross(self.fwd, target)) / d
        angle = np.math.atan2(2*self.L*sin_eta, d)

        # make signal
        drive_cmd = AckermannDriveStamped()
        drive_cmd.drive.steering_angle = angle
        
        # drive_cmd.drive.speed = copysign(1.0, d - self.parking_distance) * self.speed \
        #                 if abs(d - self.parking_distance) > self.STOP_EPS else 0.0 
        drive_cmd.drive_speed = self.speed if d > self.parking_distance else 0.0
                        # if within parking distance of center of cone stop, otherwise go to that point
        self.drive_pub.publish(drive_cmd)
        self.error_publisher()

    def error_publisher(self):
        """
        Publish the error between the car and the cone. We will view this
        with rqt_plot to plot the success of the controller
        """
        error_msg = ParkingError()
        error_msg.x_error = self.relative_x
        error_msg.y_error = self.relative_y
        error_msg.distance_error = abs(np.sqrt(self.relative_x**2+ self.relative_x**2) - self.parking_distance)
        self.error_pub.publish(error_msg)

def main(args=None):
    rclpy.init(args=args)
    pc = ParkingController()
    rclpy.spin(pc)
    rclpy.shutdown()

if __name__ == '__main__':
    main()