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

        # 원하는 IMU 목표 각도(최종적으로 로봇 IMU가 이 각도에 가깝게 되도록 유도)
        self.target_imu_angle = 90.0

        # 아직 IMU를 받기 전일 수 있으므로 None으로 초기화(필요 시 기본값 사용 가능)
        self.imu_heading = None

        # 위험도 임계값
        self.max_risk_threshold = 80.0

        # 현재 키(조향) 각도
        self.key_target_degree = 90.0

        # ROS 2 QoS 설정 및 구독
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

        # 퍼블리셔(키, 스로틀 등)
        self.key_publisher = self.create_publisher(Float32, "/actuator/key/degree", 10)
        self.thruster_publisher = self.create_publisher(
            Float32, "/actuator/thruster/percentage", 10
        )

        # 스로틀을 0으로 초기화
        thruster_msg = Float32(data=0.0)
        self.thruster_publisher.publish(thruster_msg)

        # 일정 주기로 publish
        self.create_timer(0.5, self.timer_callback)

    def timer_callback(self):
        key_msg = Float32(data=self.key_target_degree)
        self.key_publisher.publish(key_msg)

    def imu_callback(self, msg: Imu):
        # 쿼터니언 -> 오일러 변환
        quaternion = (
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        )
        roll_rad, pitch_rad, yaw_rad = euler_from_quaternion(quaternion)

        # 기존 코드: yaw_degree = 90 - degrees(yaw_rad)
        # 실제로는 로봇마다 IMU 방향 보정이 다를 수 있으니 적절히 변환
        yaw_degree = 90 - degrees(yaw_rad)

        # 현재 로봇 IMU 방위를 멤버 변수에 저장
        self.imu_heading = yaw_degree

    def lidar_callback(self, data):
        # 1) 라이다 데이터 전처리
        ranges = np.array(data.ranges)
        relevant_data = ranges[500:1500]
        relevant_data = relevant_data[
            (relevant_data != 0) & (relevant_data != float("inf"))
        ]
        # 필요하다면 뒤집기도 가능
        # relevant_data = relevant_data[::-1]

        # 2) 위험도 계산을 위한 배열 준비
        cumulative_distance = np.zeros(181)
        sample_count = np.zeros(181)
        average_distance = np.zeros(181)
        risk_values = np.zeros(181)
        risk_map = np.zeros(181)

        # 3) 거리 누적
        for i in range(len(relevant_data)):
            length = relevant_data[i]
            angle_index = round(i * 180 / len(relevant_data))
            cumulative_distance[angle_index] += length
            sample_count[angle_index] += 1

        # 4) 평균 거리
        for j in range(181):
            if sample_count[j] != 0:
                average_distance[j] = cumulative_distance[j] / sample_count[j]

        # 5) 위험도 계산(예: 135.72 * e^(-0.6109 * 거리))
        for k in range(181):
            if average_distance[k] != 0:
                risk_values[k] = 135.72 * math.exp(-0.6109 * average_distance[k])

        # 6) 위험도가 임계치 이상인 각도 주변도 위험 표시
        for k in range(181):
            if risk_values[k] >= self.max_risk_threshold:
                devide1(risk_map, k, 18)

        # 7) 안전각도(safe_angles) 구하기
        safe_angles = np.where(risk_map == 0)[0].tolist()

        # 8) 목표 각도 결정
        # IMU가 아직 업데이트 안 됐다면 기본값 90.0 사용(혹은 다른 초기값)
        current_imu = self.imu_heading if self.imu_heading is not None else 90.0

        # "IMU를 90°로 만들기" → 실제 IMU(현재_imu)가 90°와 얼마나 차이 나는지 계산
        # 한 번에 바로 90°로 가도 되지만, 여기서는 단계적으로 접근하는 예시
        heading_diff = self.target_imu_angle - current_imu  # 90 - 현재_imu
        step_factor = 0.8  # 원하는 비율만큼만 이동 (0~1 사이가 보통)
        desired_heading = current_imu + heading_diff * step_factor

        # 8-1) safe_angles가 하나 이상 있으면, desired_heading에 가장 가까운 안전 각도 선택
        if len(safe_angles) > 0:
            self.key_target_degree = float(
                min(safe_angles, key=lambda x: abs(x - desired_heading))
            )
        else:
            # 안전 각도가 전혀 없으면 임의로 135도
            # (원하면 가장 위험도가 낮은 각도(min(risk_values))를 택해도 됨)
            self.key_target_degree = 135.0

        # 9) 각도를 45~135 범위로 제한(필요 없다면 주석 처리)
        if self.key_target_degree > 135.0:
            self.key_target_degree = 135.0
        if self.key_target_degree < 45.0:
            self.key_target_degree = 45.0

        self.get_logger().info(
            f"key: {self.key_target_degree:5.1f}, IMU: {current_imu:5.1f}, safe_angles: {safe_angles}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
