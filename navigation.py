import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32


class NavigationNode(Node):
    def __init__(self):
        super().__init__("Auto_sailing")

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

    def lidar_callback(self, data):
        front_data = np.array(data.ranges[200:1800])
        valid_data = front_data[(front_data != 0) & (front_data != float("inf"))]
        min_index = np.argmin(valid_data)
        heading = np.clip(90 - 45 + (min_index * 90 / len(valid_data)), 45, 135)

        self.key_target_degree = heading
        self.get_logger().info(f"key: {self.key_target_degree:5.1f}")


def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
