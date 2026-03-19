import sys
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus

# Import your custom action
from my_robot_interfaces.action import RunVision

class BrainClient(Node):
    def __init__(self):
        super().__init__('brain_client_node')
        
        # Create the action client
        self._action_client = ActionClient(self, RunVision, 'run_perception_pipeline')
        self.get_logger().info("Brain Client initialized. Ready to dispatch tasks.")

    def send_perception_task(self, task_name: str, object_class: str = ""):
        self.get_logger().info("Waiting for Perception Dispatcher to come online...")
        
        # Check if the server is available
        if not self._action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("Perception Dispatcher not found. Is the launch file running?")
            return

        goal_msg = RunVision.Goal()
        goal_msg.task_name = task_name
        goal_msg.object_class = object_class
        
        self.get_logger().info(f"Sending goal request for task: '{task_name}', class='{object_class}'")
        
        # Send the goal and attach callbacks
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def feedback_callback(self, feedback_msg):
        """Prints live updates from the Dispatcher about what the AI is doing."""
        feedback = feedback_msg.feedback
        self.get_logger().info(f"Dispatcher Feedback: [{feedback.current_phase}]")

    def goal_response_callback(self, future):
        """Handles whether the Dispatcher accepted or rejected the task."""
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().error('Task was REJECTED by the Perception Dispatcher.')
            self._shutdown_client()
            return

        self.get_logger().info('Task accepted! Waiting for pipeline execution to finish...')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """Processes the final outcome of the vision task."""
        status = future.result().status
        result = future.result().result
        
        if status == GoalStatus.STATUS_SUCCEEDED:
            if result.success:
                self.get_logger().info("--------------------------------------------------")
                self.get_logger().info("✅ VISION TASK SUCCESSFUL")
                self.get_logger().info(f"RESULT: {result.message}")
                self.get_logger().info("--------------------------------------------------")
                
                # If this was a placement task, you can access the pose directly if 
                # your action definition includes a pose/poses field.
                # Example: if result.poses: ...
            else:
                self.get_logger().error(f"❌ Vision Task Failed: {result.message}")
        else:
            self.get_logger().error(f"🚨 Action Server Error. Status Code: {status}")

        self._shutdown_client()

    def _shutdown_client(self):
        """Cleanly shuts down the node after task completion."""
        self.get_logger().info("Task complete. Shutting down Brain Client.")
        # We use a timer to ensure the logs are printed before shutdown
        self.create_timer(0.5, lambda: sys.exit(0))

def main(args=None):
    # List of tasks your Dispatcher and Launch file are configured to handle
    valid_tasks = [
        'table_height', 
        'detect_screws', 
        'car_objects', 
        'subdoor_pose', 
        'detect_screwdriver', 
        'place_object'  # <--- NEW
    ]
    
    # Simple CLI argument check
    if len(sys.argv) < 2:
        print("Usage: ros2 run perception brain_client <task_name> [object_class]")
        sys.exit(1)

    task_name = sys.argv[1]
    # Get optional class name if provided, else empty string
    object_class = sys.argv[2] if len(sys.argv) > 2 else ''

    rclpy.init(args=args)
    brain_client = BrainClient()
    
    # Start the task
    brain_client.send_perception_task(task_name, object_class)
    
    try:
        rclpy.spin(brain_client)
    except KeyboardInterrupt:
        print("\nBrain client stopped by user.")
    finally:
        brain_client.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()