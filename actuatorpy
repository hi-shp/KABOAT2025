import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


class ActuatorControlNode(Node):
    def __init__(self):
        super().__init__("Actuator_Control")
        self.thruster_publisher = self.create_publisher(
            Float32, "/actuator/thruster/percentage", 10
        )
        thruster_msg = Float32(data=0.0)
        self.thruster_publisher.publish(thruster_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ActuatorControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
