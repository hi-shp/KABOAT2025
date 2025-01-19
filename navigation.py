import numpy as np
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu
from std_msgs.msg import Float32
from rclpy.qos import qos_profile_sensor_data
from tf_transformations import euler_from_quaternion
from math import degrees


def devide1(array, center, spread):
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
        self.target_imu_angle = 90.0
        self.imu_heading = None
        self.max_risk_threshold = 80.0
        self.key_target_degree = 90.0
        lidar_qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            durability=rclpy.qos.QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_VOLATILE,
            history=rclpy.qos.QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=5,
        )
        self.lidar_subscription = self.create_subscription(
            LaserScan, "/scan", self.lidar_callback, lidar_qos_profile
        )
        self.create_subscription(
            Imu, "/imu", self.imu_callback, qos_profile_sensor_data
        )
        self.key_publisher = self.create_publisher(Float32, "/actuator/key/degree", 10)
        self.thruster_publisher = self.create_publisher(
            Float32, "/actuator/thruster/percentage", 10
        )
        thruster_msg = Float32(data=10.0)
        self.thruster_publisher.publish(thruster_msg)
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
        yaw_degree = 90 - degrees(yaw_rad)
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
            angle_index = round(i * 180 / len(relevant_data))
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
                devide1(risk_map, k, 18)

        safe_angles = np.where(risk_map == 0)[0].tolist()
        heading_diff = self.target_imu_angle - self.imu_heading
        step_factor = 0.8
        desired_heading = self.imu_heading + heading_diff * step_factor

        def find_largest_safe_zone_center(safe_angles):
            # 연속된 구간을 탐색
            safe_zones = []
            start = None

            for i in range(len(safe_angles)):
                if start is None:
                    start = safe_angles[i]
                if (
                    i == len(safe_angles) - 1
                    or safe_angles[i] + 1 != safe_angles[i + 1]
                ):
                    end = safe_angles[i]
                    safe_zones.append((start, end))
                    start = None

            # 가장 큰 구간 선택
            if safe_zones:
                largest_zone = max(safe_zones, key=lambda x: x[1] - x[0])
                center = (largest_zone[0] + largest_zone[1]) // 2
                return center
            return None

        if len(safe_angles) > 0:
            # 가장 큰 안전 구간의 중앙값 찾기
            largest_safe_center = find_largest_safe_zone_center(safe_angles)
            if largest_safe_center is not None:
                raw_heading = float(largest_safe_center)
            else:
                raw_heading = float(
                    min(safe_angles, key=lambda x: abs(x - desired_heading))
                )
        else:
            raw_heading = 135.0

        inverted_heading = 180.0 - raw_heading

        if inverted_heading > 135.0:
            inverted_heading = 135.0
        if inverted_heading < 45.0:
            inverted_heading = 45.0

        self.key_target_degree = inverted_heading

        self.get_logger().info(
            f"key: {self.key_target_degree:5.1f}, IMU: {self.imu_heading:5.1f}, safe_angles: {safe_angles}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
