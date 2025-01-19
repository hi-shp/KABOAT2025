import numpy as np
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu
from std_msgs.msg import Float32
from rclpy.qos import qos_profile_sensor_data
from tf_transformations import euler_from_quaternion
from mechaship_interfaces.msg import RgbwLedColor
from math import degrees
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)


def set_risk_zone(array, center, spread):
    array[center] = 1
    for i in range(1, spread + 1):
        if center - i >= 0:
            array[center - i] = 1
        if center + i <= 180:
            array[center + i] = 1
    return array


class NavigationNode(Node):
    def __init__(self):
        super().__init__("Auto_sailing")
        self.imu_heading = 90.0
        self.max_risk_threshold = 70.0 #위험도 한계값
        self.key_target_degree = 90.0
        self.target_imu_angle = 90.0
        lidar_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        gps_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        imu_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.lidar_subscription = self.create_subscription(
            LaserScan, "/scan", self.lidar_callback, lidar_qos_profile
        )
        self.create_subscription(
            Imu, "/imu", self.imu_callback, qos_profile_sensor_data
        )
        self.rgbw_led_publisher = self.create_publisher(
            RgbwLedColor, "/actuator/rgbwled/color", 10
        )
        self.key_publisher = self.create_publisher(Float32, "/actuator/key/degree", 10)
        self.thruster_publisher = self.create_publisher(
            Float32, "/actuator/thruster/percentage", 10
        )
        thruster_msg = Float32(data=0.0) #속도값 (15~)
        self.thruster_publisher.publish(thruster_msg)

        color = RgbwLedColor()
        color.green = 20
        self.rgbw_led_publisher.publish(color)

        self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        key_msg = Float32(data=self.key_target_degree)
        self.key_publisher.publish(key_msg)

    def imu_callback(self, msg: Imu):
        quaternion = (
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        )
        roll_rad, pitch_rad, yaw_rad = euler_from_quaternion(quaternion)
        yaw = degrees(yaw_rad)
        if yaw <= 90.0:
            yaw_degree = 90.0 - yaw
        elif yaw > 90.0:
            yaw_degree = 450.0 - yaw

        if 0 <= yaw_degree <= 180:
            self.target_imu_angle = 90.0
        elif 180 < yaw_degree < 360:
            self.target_imu_angle = 270.0
        else:
            self.target_imu_angle = 90.0
        self.imu_heading = yaw_degree

    def lidar_callback(self, data):
        ranges = np.array(data.ranges)
        relevant_data = ranges[500:1500]
        relevant_data = relevant_data[
            (relevant_data != 0) & (relevant_data != float("inf"))
        ]
        cumulative_distance = np.zeros(181)
        sample_count = np.zeros(181)
        average_distance = np.zeros(181)
        risk_values = np.zeros(181)
        risk_map = np.zeros(181)

        for i in range(len(relevant_data)):
            length = relevant_data[i]
            angle_index = round((len(relevant_data) - 1 - i) * 180 / len(relevant_data))
            cumulative_distance[angle_index] += length
            sample_count[angle_index] += 1

        for j in range(181):
            if sample_count[j] != 0:
                average_distance[j] = cumulative_distance[j] / sample_count[j]

        for k in range(181):
            if average_distance[k] != 0:
                risk_values[k] = 135.72 * math.exp(-0.6109 * average_distance[k])

        for k in range(181):
            if risk_values[k] >= self.max_risk_threshold:
                set_risk_zone(risk_map, k, 25) #장애물 인식 너비

        safe_angles = np.where(risk_map == 0)[0].tolist()
        heading_diff = float(self.target_imu_angle - self.imu_heading)
        step_factor = 1
        if self.target_imu_angle == 90.0:
            desired_heading = self.target_imu_angle + heading_diff * step_factor
        else:
            desired_heading = self.target_imu_angle + heading_diff * step_factor - 180.0

        if len(safe_angles) > 0:
            heading = float(min(safe_angles, key=lambda x: abs(x - desired_heading)))
        else:
            heading = 45.0

        if heading > 135.0:
            heading = 135.0
        if heading < 45.0:
            heading = 45.0

        self.key_target_degree = heading

        self.get_logger().info(
            f"key: {self.key_target_degree:5.1f}, IMU: {self.imu_heading:5.1f}, Target IMU: {self.target_imu_angle},"
        )


def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
