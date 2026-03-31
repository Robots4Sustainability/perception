import sys
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus

from my_robot_interfaces.action import RunVision

VALID_TASKS = {
    'table_height':        {'usage': "ros2 run perception brain_client table_height"},
    'detect_screws':       {'usage': "ros2 run perception brain_client detect_screws"},
    'car_objects':         {'usage': "ros2 run perception brain_client car_objects <object_class>  (e.g. motor, motor_grip)"},
    'subdoor_pose':        {'usage': "ros2 run perception brain_client subdoor_pose"},
    'detect_screwdriver':  {'usage': "ros2 run perception brain_client detect_screwdriver"},
    'place_object':        {'usage': "ros2 run perception brain_client place_object <radius_in_meters>  (e.g. 0.05)"},
}

def print_usage_and_exit(invalid_task=None):
    if invalid_task:
        print(f"\n❌ Unknown task: '{invalid_task}'")
    print("\nAvailable tasks:")
    for task, info in VALID_TASKS.items():
        print(f"  {task:<22} →  {info['usage']}")
    print()
    sys.exit(1)

def parse_args():
    """Parse CLI args based on which task is being requested."""
    if len(sys.argv) < 2:
        print_usage_and_exit()

    task_name = sys.argv[1]

    if task_name not in VALID_TASKS:
        print_usage_and_exit(invalid_task=task_name)

    object_class = ''
    radius = 0.0

    if task_name == 'car_objects':
        if len(sys.argv) < 3:
            print(f"\n❌ 'car_objects' requires an object_class argument.")
            print(f"   Usage: {VALID_TASKS['car_objects']['usage']}\n")
            sys.exit(1)
        object_class = sys.argv[2]

    elif task_name == 'place_object':
        if len(sys.argv) < 3:
            print(f"\n❌ 'place_object' requires a radius argument (in meters).")
            print(f"   Usage: {VALID_TASKS['place_object']['usage']}\n")
            sys.exit(1)
        try:
            radius = float(sys.argv[2])
        except ValueError:
            print(f"\n❌ Invalid radius '{sys.argv[2]}'. Must be a float (e.g. 0.05 for 5cm).\n")
            sys.exit(1)

    return task_name, object_class, radius


class BrainClient(Node):
    def __init__(self):
        super().__init__('brain_client_node')
        self._action_client = ActionClient(self, RunVision, 'run_perception_pipeline')
        self.get_logger().info("Brain Client initialized. Ready to dispatch tasks.")

    def send_perception_task(self, task_name: str, object_class: str = "", radius: float = 0.0):
        self.get_logger().info("Waiting for Perception Dispatcher to come online...")

        if not self._action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("Perception Dispatcher not found. Is the launch file running?")
            return

        goal_msg = RunVision.Goal()
        goal_msg.task_name = task_name
        goal_msg.object_class = object_class
        goal_msg.radius = radius

        log = f"Sending goal request for task: '{task_name}'"
        if object_class:
            log += f", class='{object_class}'"
        if radius > 0.0:
            log += f", radius={radius:.4f}m"
        self.get_logger().info(log)

        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def feedback_callback(self, feedback_msg):
        self.get_logger().info(f"Dispatcher Feedback: [{feedback_msg.feedback.current_phase}]")

    def goal_response_callback(self, future):
        try:
            self._goal_handle = future.result()
            if not self._goal_handle.accepted:
                self.get_logger().error('Goal rejected')
                return
            self.get_logger().info('Goal accepted!')
            self.get_logger().info('Requesting result...')
            self._get_result_future = self._goal_handle.get_result_async()
            self._get_result_future.add_done_callback(self.get_result_callback)
        except Exception as e:
            self.get_logger().error(f"Logic Error in Response Callback: {e}")

    def get_result_callback(self, future):
        try:
            action_result = future.result()
            status = action_result.status
            result = action_result.result

            if status == 4:  # STATUS_SUCCEEDED
                self.get_logger().info("--------------------------------------------------")
                self.get_logger().info("✅ VISION TASK SUCCESSFUL")
                self.get_logger().info(f"DATA: {result.message}")
                if result.poses:
                    p = result.poses[0].pose.position
                    self.get_logger().info(f"Raw Coordinates: x={p.x:.3f}, y={p.y:.3f}, z={p.z:.3f}")
                self.get_logger().info("--------------------------------------------------")
            else:
                self.get_logger().error(f"❌ Task failed with status code: {status}")
                self.get_logger().error(f"   Message: {result.message}")

        except Exception as e:
            self.get_logger().error(f"Error printing result: {e}")
        finally:
            self._shutdown_client()

    def _shutdown_client(self):
        self.get_logger().info("Task complete. Shutting down Brain Client.")
        self.create_timer(0.5, lambda: sys.exit(0))


def main(args=None):
    task_name, object_class, radius = parse_args()

    rclpy.init(args=args)
    brain_client = BrainClient()
    brain_client.send_perception_task(task_name, object_class, radius)

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