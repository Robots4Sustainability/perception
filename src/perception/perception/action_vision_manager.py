import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import SetBool
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import statistics

from my_robot_interfaces.action import RunVision

class VisionManager(Node):
    def __init__(self):
        super().__init__('vision_manager')
        
        # We use a ReentrantCallbackGroup to allow the Action Loop and the 
        # Subscriber Callback to run in parallel on the MultiThreadedExecutor
        self.group = ReentrantCallbackGroup()

        # --- 1. ACTION SERVER ---
        self._action_server = ActionServer(
            self,
            RunVision,
            'run_vision_pipeline',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.group
        )
        
        # --- 2. SERVICE CLIENTS (Talks to YOLO and Pose nodes) ---
        self.yolo_client = self.create_client(SetBool, '/toggle_yolo', callback_group=self.group)
        self.pose_client = self.create_client(SetBool, '/toggle_pose', callback_group=self.group)
        
        # --- 3. STATE MANAGEMENT ---
        self.captured_poses = []
        self.collection_active = False  # The "Gate" flag
        
        # --- 4. SUBSCRIBER ---
        # Reliable QoS ensures we don't miss packets if they are sent correctly
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/object_pose',
            self.pose_callback,
            qos_profile,
            callback_group=self.group
        )

        self.vis_pub = self.create_publisher(
            PoseStamped, 
            '/visualized_average_pose', 
            10
        )
        
        self.get_logger().info("Vision Manager Ready. Initializing sensors to OFF state...")
        
        # --- 5. INITIAL CLEANUP TIMER ---
        # We run this ONCE to ensure sensors start in a clean "Standby" state.
        self.init_timer = self.create_timer(1.0, self.initial_cleanup)

    async def initial_cleanup(self):
        """Forces connected nodes to sleep on startup so we start fresh."""
        
        # CRITICAL FIX: Cancel the timer so this function NEVER runs again
        if self.init_timer:
            self.init_timer.cancel()
            self.init_timer = None

        self.get_logger().info("System Startup: Ensuring sensors are in STANDBY mode...")
        await self.set_nodes_state(False)

    def pose_callback(self, msg):
        """
        Only process data if the Action is running (collection_active is True).
        Otherwise, ignore the noise to prevent log spam and dirty data.
        """
        if not self.collection_active:
            return  # IGNORE BACKGROUND NOISE

        self.get_logger().info(f"Received Pose: x={msg.pose.position.x:.2f}")
        self.captured_poses.append(msg)

    def goal_callback(self, goal_request):
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Goal Received. Opening the Gate...')
        
        # 1. Reset Buffer for new run
        self.captured_poses = []
        result = RunVision.Result()
        feedback = RunVision.Feedback()
        duration = goal_handle.request.duration_seconds
        
        # 2. Wake up nodes
        if not await self.set_nodes_state(True):
            goal_handle.abort()
            result.success = False
            result.message = "Failed to wake sensors"
            return result
        
        # 3. Enable Collection (Open Gate)
        self.collection_active = True  

        # 4. Run Timer Loop
        start_time = time.time()
        feedback.status = "Collecting Poses"
        
        while (time.time() - start_time) < duration:
            # Check Cancel
            if goal_handle.is_cancel_requested:
                self.collection_active = False # Close gate
                await self.set_nodes_state(False)
                goal_handle.canceled()
                result.success = False
                result.message = "Canceled"
                return result
            
            # Publish Feedback
            feedback.time_elapsed = time.time() - start_time
            goal_handle.publish_feedback(feedback)
            time.sleep(0.1)

        # 5. Disable Collection (Close Gate)
        self.collection_active = False  
        
        # 6. Shut Down Sensors
        await self.set_nodes_state(False) 

        # 7. Process Data
        if len(self.captured_poses) > 0:
            final_pose = self.calculate_average_pose(self.captured_poses)
            
            # Publish to RViz so we can see the result
            self.vis_pub.publish(final_pose)
            self.get_logger().info("Published final median pose to /visualized_average_pose")
            
            result.success = True
            result.message = f"Success. Calculated Median of {len(self.captured_poses)} frames."
            result.pose = final_pose
            goal_handle.succeed()
        else:
            result.success = False
            result.message = "Time finished, but no objects were detected."
            self.get_logger().warn("Finished with 0 poses.")
            goal_handle.abort()

        return result

    def calculate_average_pose(self, poses):
        """
        Calculates the 'Truncated Mean' (Interquartile Mean).
        1. Sorts data.
        2. Removes top 25% and bottom 25% (outliers).
        3. Averages the remaining middle 50%.
        """
        if not poses: return PoseStamped()
        
        # Helper function to get truncated mean of a list of numbers
        def get_truncated_mean(data):
            if not data: return 0.0
            n = len(data)
            if n < 3: return statistics.mean(data) # Too few to truncate
            
            sorted_data = sorted(data)
            # Remove top and bottom 25%
            cut_amount = int(n * 0.25)
            # Slice the middle
            middle_data = sorted_data[cut_amount : n - cut_amount]
            
            return statistics.mean(middle_data)

        # 1. Calculate Truncated Mean for Position
        final_x = get_truncated_mean([p.pose.position.x for p in poses])
        final_y = get_truncated_mean([p.pose.position.y for p in poses])
        final_z = get_truncated_mean([p.pose.position.z for p in poses])

        # 2. Find the orientation from the pose closest to this new calculated position
        # (We still shouldn't average quaternions simply, so we pick the best representative)
        best_idx = 0
        min_dist = float('inf')
        
        for i, p in enumerate(poses):
            dist = (p.pose.position.x - final_x)**2 + \
                   (p.pose.position.y - final_y)**2 + \
                   (p.pose.position.z - final_z)**2
            if dist < min_dist:
                min_dist = dist
                best_idx = i

        # 3. Construct Final Pose
        final_pose = PoseStamped()
        final_pose.header = poses[best_idx].header
        final_pose.header.stamp = self.get_clock().now().to_msg()
        
        final_pose.pose.position.x = final_x
        final_pose.pose.position.y = final_y
        final_pose.pose.position.z = final_z
        final_pose.pose.orientation = poses[best_idx].pose.orientation
        
        return final_pose

    async def set_nodes_state(self, active: bool):
        """Helper to call services asynchronously."""
        req = SetBool.Request()
        req.data = active
        
        # Small timeout check to avoid blocking if nodes aren't up
        if not self.yolo_client.wait_for_service(timeout_sec=1.0):
             self.get_logger().warn("YOLO Service not available")
             return False
        if not self.pose_client.wait_for_service(timeout_sec=1.0):
             self.get_logger().warn("Pose Service not available")
             return False

        future_yolo = self.yolo_client.call_async(req)
        future_pose = self.pose_client.call_async(req)
        
        try:
            await future_yolo
            await future_pose
            return True
        except Exception as e:
            self.get_logger().error(f"Service toggle failed: {e}")
            return False

def main(args=None):
    rclpy.init(args=args)
    node = VisionManager()
    
    # CRITICAL: MultiThreadedExecutor is required for Callbacks to run 
    # while the Action Loop is active.
    executor = MultiThreadedExecutor()
    rclpy.spin(node, executor=executor)
    rclpy.shutdown()

if __name__ == '__main__':
    main()