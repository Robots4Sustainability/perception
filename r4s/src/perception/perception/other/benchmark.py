#!/usr/bin/env python3
"""
Simple latency benchmark for perception package
Works with the two-node setup (yolo_node + pose_node)
"""

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import time
from collections import deque
import statistics


class SimpleBenchmark(Node):
    
    def __init__(self):
        super().__init__('benchmark_node')
        
        # Storage for timing data
        self.detections_received = deque(maxlen=300)
        self.poses_received = deque(maxlen=300)
        
        # Latency tracking
        self.detection_latencies = deque(maxlen=100)
        self.pose_latencies = deque(maxlen=100)
        self.e2e_latencies = deque(maxlen=100)
        
        # Track message timing
        self.last_image_time = None
        self.detection_count = 0
        self.pose_count = 0
        
        self.get_logger().info("Subscribing to topics...")
        
        # Subscribe to raw image (for baseline timing)
        self.image_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10)
        
        # Subscribe to detections
        self.det_sub = self.create_subscription(
            Detection2DArray, '/detections', self.detection_callback, 10)
        
        # Subscribe to poses (end-to-end latency)
        self.pose_sub = self.create_subscription(
            PoseStamped, '/object_pose', self.pose_callback, 10)
        
        # Subscribe to annotated image
        self.annotated_sub = self.create_subscription(
            Image, '/annotated_image', self.annotated_callback, 10)
        
        # Timer for reporting
        self.timer = self.create_timer(5.0, self.report_stats)
        
        self.get_logger().info("Benchmark started. Listening for messages...")
    
    
    def image_callback(self, msg):
        """Track when raw image arrives."""
        self.last_image_time = time.time()
    
    
    def detection_callback(self, msg):
        """Track detections."""
        self.detection_count += 1
        self.detections_received.append(time.time())
        
        if self.last_image_time and len(msg.detections) > 0:
            latency = (time.time() - self.last_image_time) * 1000
            self.detection_latencies.append(latency)
    
    
    def pose_callback(self, msg):
        """Track poses (end-to-end latency)."""
        self.pose_count += 1
        self.poses_received.append(time.time())
        
        if self.last_image_time:
            latency = (time.time() - self.last_image_time) * 1000
            self.e2e_latencies.append(latency)
    
    
    def annotated_callback(self, msg):
        """Just for confirmation node is running."""
        pass
    
    
    def report_stats(self):
        """Print statistics every 5 seconds."""
        
        print("\n" + "="*70)
        print("LATENCY BENCHMARK - PERCEPTION PACKAGE")
        print("="*70)
        
        # Detection latency
        if self.detection_latencies:
            det_list = list(self.detection_latencies)
            print("\n📊 DETECTION LATENCY (Image → Detection Output)")
            print(f"   Count:    {len(det_list)} detections")
            print(f"   Mean:     {statistics.mean(det_list):>8.2f} ms")
            print(f"   Median:   {statistics.median(det_list):>8.2f} ms")
            print(f"   Min:      {min(det_list):>8.2f} ms")
            print(f"   Max:      {max(det_list):>8.2f} ms")
            if len(det_list) > 1:
                print(f"   StdDev:   {statistics.stdev(det_list):>8.2f} ms")
        else:
            print("\n⚠️  No detection data yet. Check if yolo_node is running.")
        
        # End-to-end latency
        if self.e2e_latencies:
            e2e_list = list(self.e2e_latencies)
            print("\n🔍 END-TO-END LATENCY (Image → Pose Output)")
            print(f"   Count:    {len(e2e_list)} poses")
            print(f"   Mean:     {statistics.mean(e2e_list):>8.2f} ms")
            print(f"   Median:   {statistics.median(e2e_list):>8.2f} ms")
            print(f"   Min:      {min(e2e_list):>8.2f} ms")
            print(f"   Max:      {max(e2e_list):>8.2f} ms")
            if len(e2e_list) > 1:
                print(f"   StdDev:   {statistics.stdev(e2e_list):>8.2f} ms")
        else:
            print("\n⚠️  No pose data yet. Check if pose_node is running.")
        
        # Throughput
        if self.detections_received:
            time_span = self.detections_received[-1] - self.detections_received[0]
            if time_span > 0:
                throughput = len(self.detections_received) / time_span
                print(f"\n⚡ THROUGHPUT: {throughput:.1f} fps")
        
        print("="*70 + "\n")


def main(args=None):
    rclpy.init(args=args)
    benchmark = SimpleBenchmark()
    
    try:
        rclpy.spin(benchmark)
    except KeyboardInterrupt:
        print("\n\nBenchmark stopped by user.")
    finally:
        benchmark.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()