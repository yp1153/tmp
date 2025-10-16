#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import sensor_msgs_py.point_cloud2 as pc2
from scipy.spatial import cKDTree
import math

class Vslam2DMap(Node):
    def __init__(self):
        super().__init__('vslam_2d_map')
        self.sub_cloud = self.create_subscription(
            PointCloud2,
            '/visual_slam/vis/landmarks_cloud',
            self.cloud_callback,
            10
        )
        self.sub_odom = self.create_subscription(
            Odometry,
            '/visual_slam/tracking/odometry',
            self.odom_callback,
            10
        )

        self.latest_pose = None
        self.received_cloud = False

        self.get_logger().info("Waiting for /visual_slam/vis/landmarks_cloud and /visual_slam/tracking/odometry...")

    def odom_callback(self, msg: Odometry):
        self.latest_pose = msg.pose.pose

    def cloud_callback(self, msg: PointCloud2):
        if self.received_cloud:
            return
        self.received_cloud = True

        # --- PointCloud2 → numpy array ---
        points_list = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        if len(points_list) == 0:
            self.get_logger().warn("No valid points in PointCloud2!")
            rclpy.shutdown()
            return

        points = np.array([[p[0], p[1], p[2]] for p in points_list], dtype=np.float32)

        # --- z > k 제거 ---
        k = 0.3
        points = points[points[:, 2] <= k

        ]

        # --- z=0 평면 투영 ---
        points[:, 2] = 0.0

        # --- 이상치 제거 ---
        points = self.remove_outliers(points)

        self.get_logger().info(f"Points after outlier removal: {len(points)}")
        self.visualize(points)

    # ---------------- 이상치 제거 ----------------
    def remove_outliers(self, points, method='stat_radius', nb_neighbors=5, std_ratio=2.0, radius=0.5, min_neighbors=3):
        """
        method:
            'stat' - 통계적 이상치 제거
            'radius' - 반경 기반 제거
            'stat_radius' - 두 방법 결합
        """
        filtered = points

        if method in ['stat', 'stat_radius']:
            # 통계적 이상치 제거
            tree = cKDTree(filtered[:, :2])
            distances = []
            for i, pt in enumerate(filtered[:, :2]):
                dists, idx = tree.query(pt, k=nb_neighbors+1)
                distances.append(np.mean(dists[1:]))
            distances = np.array(distances)
            mean = distances.mean()
            std = distances.std()
            mask = distances < mean + std_ratio * std
            filtered = filtered[mask]

        if method in ['radius', 'stat_radius']:
            # 반경 기반 제거
            tree = cKDTree(filtered[:, :2])
            mask = []
            for pt in filtered[:, :2]:
                idx = tree.query_ball_point(pt, radius)
                mask.append(len(idx) > min_neighbors)
            filtered = filtered[mask]

        return filtered

# ---------------- 시각화 ----------------
    def visualize(self, points):
        # --- Figure 1: 기존 시각화 (odometry 포함) ---
        plt.figure(1, figsize=(8, 8))
        plt.scatter(points[:, 0], points[:, 1], s=2, c='black', alpha=1.0, label='Landmarks')

        # odometry 표시
        if self.latest_pose:
                pos = self.latest_pose.position
                ori = self.latest_pose.orientation

                x, y = pos.x, pos.y
                yaw = math.atan2(
                        2.0 * (ori.w * ori.z + ori.x * ori.y),
                        1.0 - 2.0 * (ori.y**2 + ori.z**2)
                )
                dx = 0.3 * math.cos(yaw)
                dy = 0.3 * math.sin(yaw)

                plt.arrow(x, y, dx, dy, color='red', width=0.01, label='Odometry Direction')
                plt.scatter(x, y, c='blue', s=30, label='Odometry Position')
        else:
                self.get_logger().warn("No odometry received. Only landmarks displayed.")

        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title('2D VSLAM Map (Outlier Removed)')
        plt.legend()
        plt.axis('equal')
        plt.grid(True)

        # --- Figure 2: Landmarks만 표시 ---
        plt.figure(2, figsize=(8, 8))
        plt.scatter(points[:, 0], points[:, 1], s=2, c='gray', alpha=0.6, label='Landmarks Only')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title('Landmarks Only')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()

        plt.show()  # Figure 1과 Figure 2 동시에 화면에 표시

        self.get_logger().info("Visualization complete. Exiting...")
        rclpy.shutdown()


def main():
    rclpy.init()
    node = Vslam2DMap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
