#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np
from numpy.linalg import norm

from vs_msgs.msg import ConeLocation, ParkingError
from ackermann_msgs.msg import AckermannDriveStamped

class ParkingController(Node):
    """
    A controller for parking in front of a cone.
    Listens for a relative cone location and publishes control commands.
    Can be used in the simulator and on the real robot.
    """
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
        self.epsilon = 1e-1

        self.get_logger().info("Parking Controller Initialized")

    def relative_cone_callback(self, msg):
        self.relative_x = msg.x_pos
        self.relative_y = msg.y_pos
        # get target point
        cone = np.array([msg.x_pos, msg.y_pos])
        norm_cone = norm(cone)
        s_cone = np.sign(msg.y_pos)*norm(np.cross(self.fwd, cone)) / norm_cone
        c_cone = np.dot(self.fwd, cone) / norm_cone
        target = np.array([msg.x_pos - self.parking_distance*s_cone,
                           msg.y_pos - self.parking_distance*c_cone]).T
        
        # steer
        d = np.linalg.norm(target)
        sin_eta = np.sign(target[1])*norm(np.cross(self.fwd, target)) / d
        angle = -np.math.atan2(2*self.L*sin_eta, d)

        # make signal
        drive_cmd = AckermannDriveStamped()
        drive_cmd.drive.steering_angle = angle
        drive_cmd.drive.speed = 0.0 if d < self.epsilon else 1.0
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
        error_msg.distance_error = np.sqrt(self.relative_x**2+ self.relative_x**2)
        self.error_pub.publish(error_msg)

def main(args=None):
    rclpy.init(args=args)
    pc = ParkingController()
    rclpy.spin(pc)
    rclpy.shutdown()

if __name__ == '__main__':
    main()