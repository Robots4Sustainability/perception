import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.signals import SignalHandlerOptions
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32

from lifecycle_msgs.srv import ChangeState
from lifecycle_msgs.msg import Transition

from my_robot_interfaces.action import RunVision
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

import asyncio
import threading

class PerceptionDispatcher(Node):
    def __init__(self):
        super().__init__('perception_pipeline_node')
        self.group = ReentrantCallbackGroup()

        # ==========================================
        # LAZY LOADING PARAMETER
        # ==========================================
        # lazy_loading = False (default): Pre-configure all nodes at startup.
        #   Models are loaded into VRAM once and stay there.
        #   Tasks only toggle activate/deactivate. Fast task execution.
        #
        # lazy_loading = True: No pre-configuration at startup.
        #   Each task does a full configure->activate cycle on demand,
        #   then deactivate->cleanup when done. Models are fully unloaded
        #   from VRAM between tasks. Slower per-task but saves VRAM.
        self.declare_parameter('lazy_loading', False)
        self._lazy = self.get_parameter('lazy_loading').get_parameter_value().bool_value
        self.get_logger().info(
            f"Lazy loading mode: {'ENABLED (low VRAM)' if self._lazy else 'DISABLED (pre-load all)'}"
        )

        # Action Server setup
        self._action_server = ActionServer(
            self,
            RunVision,
            'run_perception_pipeline',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.group
        )

        # ==========================================
        # 1. LIFECYCLE CLIENTS (Isolated Pipelines)
        # ==========================================
        self.table_lifecycle_client = self.create_client(ChangeState, '/table_height_estimator/change_state', callback_group=self.group)
        
        # Pipeline A: Screws
        self.yolo_screw_client = self.create_client(ChangeState, '/yolo_screw_node/change_state', callback_group=self.group)
        self.cropper_screw_client = self.create_client(ChangeState, '/cropper_screw_node/change_state', callback_group=self.group)
        
        # Pipeline B: Car Objects
        self.yolo_car_client = self.create_client(ChangeState, '/yolo_car_node/change_state', callback_group=self.group)
        self.cropper_car_client = self.create_client(ChangeState, '/cropper_car_node/change_state', callback_group=self.group)

        # OBB Pipeline: Screwdriver
        self.obb_object_lifecycle_client = self.create_client(ChangeState, '/obb_detector_node/change_state', callback_group=self.group)
        self.obb_cropper_lifecycle_client = self.create_client(ChangeState, '/point_obb_cloud_cropper_node/change_state', callback_group=self.group)

        # Subdoor Pose Estimation
        self.subdoor_lifecycle_client = self.create_client(ChangeState, '/subdoor_pose_estimator/change_state', callback_group=self.group)

        # Place Object (Table Segmentation)
        self.place_object_client = self.create_client(ChangeState, '/place_object_node/change_state', callback_group=self.group)

        # ==========================================
        # 2. PERSISTENT STATE STORAGE
        # ==========================================
        self.vision_data = {
            'screw': {'pose': None, 'radius': None},
            'car object': {'pose': None, 'radius': None},
            'table': None,
            'table_pose': None,
            'subdoor': None,
            'screwdriver': None,
            'place_pose': None
        }

        # ==========================================
        # 3. PERSISTENT SUBSCRIBERS 
        # ==========================================
        self.create_subscription(PoseStamped, '/screw/pose', self.screw_pose_cb, 10, callback_group=self.group)
        self.create_subscription(Float32, '/screw/radius', self.screw_rad_cb, 10, callback_group=self.group)
        self.create_subscription(PoseStamped, '/car/pose', self.car_pose_cb, 10, callback_group=self.group)
        self.create_subscription(Float32, '/car/radius', self.car_rad_cb, 10, callback_group=self.group)
        self.create_subscription(Float32, '/table_height_value', self.table_cb, 10, callback_group=self.group)
        self.create_subscription(MarkerArray, '/body_markers', self.subdoor_cb, 10, callback_group=self.group)
        self.create_subscription(PoseStamped, '/object_pose_screwdriver', self.screwdriver_cb, 10, callback_group=self.group)
        self.create_subscription(PoseStamped, '/perception/table_pose', self.table_pose_cb, 10, callback_group=self.group)
        self.create_subscription(PoseStamped, '/perception/target_place_pose', self.place_pose_cb, 10, callback_group=self.group)

        self.get_logger().info("Perception Dispatcher initializing...")
        threading.Thread(target=self._run_startup_routine, daemon=True).start()

    # --- SUBSCRIBER CALLBACKS ---
    def screw_pose_cb(self, msg): self.vision_data['screw']['pose'] = msg
    def screw_rad_cb(self, msg): self.vision_data['screw']['radius'] = msg.data
    def car_pose_cb(self, msg): self.vision_data['car object']['pose'] = msg
    def car_rad_cb(self, msg): self.vision_data['car object']['radius'] = msg.data
    def screwdriver_cb(self, msg): self.vision_data['screwdriver'] = msg
    def place_pose_cb(self, msg): self.vision_data['place_pose'] = msg

    def table_cb(self, msg: Float32):
        self.vision_data['table'] = msg.data

    def table_pose_cb(self, msg): 
        self.vision_data['table_pose'] = msg

    def subdoor_cb(self, msg: MarkerArray):
        if msg.markers and msg.markers[0].ns == "corners":
            self.vision_data['subdoor'] = msg.markers

    # --- STARTUP ROUTINE ---
    def _run_startup_routine(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            if self._lazy:
                # In lazy mode: nodes stay unconfigured. Nothing to do at startup.
                self.get_logger().info("======================================================")
                self.get_logger().info("LAZY MODE: Skipping startup pre-configuration.")
                self.get_logger().info("Models will be loaded on demand per task.")
                self.get_logger().info("Perception Dispatcher is READY for tasks.")
                self.get_logger().info("======================================================")
            else:
                # In eager mode: configure all nodes upfront so tasks only toggle activation.
                all_successful = loop.run_until_complete(self._startup_all())
                if all_successful:
                    self.get_logger().info("======================================================")
                    self.get_logger().info("Startup routine completed. ALL nodes pre-configured.")
                    self.get_logger().info("Perception Dispatcher is READY for tasks.")
                    self.get_logger().info("======================================================")
                else:
                    self.get_logger().error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    self.get_logger().error("STARTUP PARTIALLY FAILED. Some nodes did not configure.")
                    self.get_logger().error("Check the logs above for specific node failures.")
                    self.get_logger().error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        finally:
            loop.close()

    async def _startup_all(self):
        results = await asyncio.gather(
            self._configure_and_log(self.table_lifecycle_client, 'Table Height Node'),
            self._configure_and_log(self.yolo_screw_client, 'YOLO Screw Node'),
            self._configure_and_log(self.cropper_screw_client, 'Cropper Screw Node'),
            self._configure_and_log(self.yolo_car_client, 'YOLO Car Node'),
            self._configure_and_log(self.cropper_car_client, 'Cropper Car Node'),
            self._configure_and_log(self.subdoor_lifecycle_client, 'Subdoor Pose Node'),
            self._configure_and_log(self.obb_object_lifecycle_client, 'OBB Detector Node'),
            self._configure_and_log(self.obb_cropper_lifecycle_client, 'OBB Cropper Node'),
            self._configure_and_log(self.place_object_client, 'Place Object Node'),
        )
        return all(results)

    async def _configure_and_log(self, client, node_name: str):
        self.get_logger().info(f"Pre-configuring {node_name}...")
        success = await self._change_lifecycle_state(client, Transition.TRANSITION_CONFIGURE)
        if success:
            self.get_logger().info(f"SUCCESS: {node_name} Configured.")
        else:
            self.get_logger().error(f"FAILURE: Could not configure {node_name}.")
        return success

    async def _change_lifecycle_state(self, client, transition_id):
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f"Service {client.srv_name} not available.")
            return False
            
        req = ChangeState.Request()
        req.transition.id = transition_id
        future = client.call_async(req)
        
        while not future.done():
            await asyncio.sleep(0.05)
            
        result = future.result()
        return result.success if result is not None else False

    # ==========================================
    # LIFECYCLE HELPERS — LAZY vs EAGER
    # ==========================================

    async def _node_start(self, client, node_name: str) -> bool:
        """Bring a node to Active state.
        
        Eager mode: node is already Configured, just activate it.
        Lazy mode:  node is Unconfigured, so configure it first (loads model
                    into VRAM), then activate it.
        """
        if self._lazy:
            self.get_logger().info(f"[LAZY] Configuring (loading) {node_name}...")
            if not await self._change_lifecycle_state(client, Transition.TRANSITION_CONFIGURE):
                self.get_logger().error(f"Failed to configure {node_name}.")
                return False
        return await self._change_lifecycle_state(client, Transition.TRANSITION_ACTIVATE)

    async def _node_stop(self, client, node_name: str):
        """Bring a node back to its resting state.
        
        Eager mode: deactivate only — model stays in VRAM, ready for next task.
        Lazy mode:  deactivate then cleanup — model is fully unloaded from VRAM.
        """
        await self._change_lifecycle_state(client, Transition.TRANSITION_DEACTIVATE)
        if self._lazy:
            self.get_logger().info(f"[LAZY] Cleaning up (unloading) {node_name}...")
            await self._change_lifecycle_state(client, Transition.TRANSITION_CLEANUP)

    # --- ACTION SERVER CALLBACKS ---
    def goal_callback(self, goal_request):
        self.get_logger().info(f"Received request for task: {goal_request.task_name}")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().warn("Task canceled by client.")
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        task = goal_handle.request.task_name
        self.get_logger().info(f"Executing task: {task}...")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = RunVision.Result()
        result.success = False

        try:
            if task == 'table_height':
                result = loop.run_until_complete(self.handle_table_height(goal_handle))
            elif task == 'detect_screws':
                result = loop.run_until_complete(self.handle_detect_yolo(goal_handle, self.yolo_screw_client, self.cropper_screw_client, "screw"))
            elif task == 'car_objects':
                result = loop.run_until_complete(self.handle_detect_yolo(goal_handle, self.yolo_car_client, self.cropper_car_client, "car object"))
            elif task == 'subdoor_pose':
                result = loop.run_until_complete(self.handle_subdoor_pose(goal_handle))
            elif task == 'detect_screwdriver':
                result = loop.run_until_complete(self.handle_detect_screwdriver(goal_handle))
            elif task == 'place_object':
                result = loop.run_until_complete(self.handle_place_object(goal_handle))
            else:
                result.message = f"Unknown task: {task}"
        except Exception as e:
            self.get_logger().error(f"Task failed with error: {e}")
            result.success = False
            result.message = str(e) if str(e) else type(e).__name__
        finally:
            loop.close()

        if result.success:
            goal_handle.succeed()
        else:
            try:
                goal_handle.abort()
            except Exception:
                pass 

        return result

    # --- TASK HANDLERS ---
    
    async def handle_table_height(self, goal_handle):
        feedback_msg, result_msg = RunVision.Feedback(), RunVision.Result()
        self.vision_data['table_pose'] = None

        feedback_msg.current_phase = "Activating Table Estimator..."
        goal_handle.publish_feedback(feedback_msg)
        await self._node_start(self.table_lifecycle_client, 'Table Height Node')

        feedback_msg.current_phase = "Waiting for table pose..."
        goal_handle.publish_feedback(feedback_msg)

        for _ in range(150):
            if self.vision_data['table_pose'] is not None: break
            await asyncio.sleep(0.1)

        await self._node_stop(self.table_lifecycle_client, 'Table Height Node')

        pose = self.vision_data['table_pose']
        if pose is not None:
            result_msg.success = True
            x, y, z = pose.pose.position.x, pose.pose.position.y, pose.pose.position.z
            result_msg.message = f"Table Pose Found: [x: {x:.3f}, y: {y:.3f}, z: {z:.3f} (Height)]"
        else:
            result_msg.success, result_msg.message = False, "Vision processing timed out."
        
        return result_msg

    async def handle_detect_yolo(self, goal_handle, yolo_client, cropper_client, object_name):
        feedback_msg, result_msg = RunVision.Feedback(), RunVision.Result()

        requested_class = goal_handle.request.object_class
        
        self.vision_data[object_name]['pose'] = None
        self.vision_data[object_name]['radius'] = None

        if requested_class=='motor' or requested_class=="motor_grip":
            model_type='fine_tuned'
        else:
            model_type='unit'

        # --- DYNAMIC PARAMETER OVERRIDE ---
        # In lazy mode, parameters must be set while the node is still in the
        # Configured (inactive) state — i.e. AFTER configure but BEFORE activate.
        # We therefore call _set_remote_parameters before _node_start here.
        # In eager mode nodes are already configured, so this works identically.
        if requested_class:
            self.get_logger().info(f"Targeting specific class for {object_name}: '{requested_class}'")
            
            yolo_ns = yolo_client.srv_name.replace('/change_state', '')
            await self._set_remote_parameters(yolo_ns, {
                'model_type': ParameterValue(
                    type=ParameterType.PARAMETER_STRING, 
                    string_value=model_type,
                )
            })

            cropper_ns = cropper_client.srv_name.replace('/change_state', '')
            await self._set_remote_parameters(cropper_ns, {
                'target_class': ParameterValue(
                    type=ParameterType.PARAMETER_STRING, 
                    string_value=requested_class
                )
            })
        else:
            cropper_ns = cropper_client.srv_name.replace('/change_state', '')
            await self._set_remote_parameters(cropper_ns, {
                'target_class': ParameterValue(
                    type=ParameterType.PARAMETER_STRING, 
                    string_value=''
                )
            })

        # --- LIFECYCLE ACTIVATION ---
        feedback_msg.current_phase = f"Activating Pipeline ({object_name})..."
        goal_handle.publish_feedback(feedback_msg)

        yolo_success = await self._node_start(yolo_client, f'YOLO {object_name}')
        cropper_success = await self._node_start(cropper_client, f'Cropper {object_name}')

        if not (yolo_success and cropper_success):
            result_msg.success, result_msg.message = False, f"Failed to activate {object_name} pipeline."
            return result_msg

        # --- WAIT FOR VISION DATA ---
        display_label = requested_class if requested_class else object_name
        feedback_msg.current_phase = f"Waiting for {display_label} detection..."
        goal_handle.publish_feedback(feedback_msg)

        for _ in range(150):
            if self.vision_data[object_name]['pose'] is not None:
                break
            await asyncio.sleep(0.1)

        # --- DEACTIVATION / CLEANUP ---
        feedback_msg.current_phase = f"Deactivating Pipeline ({object_name})..."
        goal_handle.publish_feedback(feedback_msg)
        
        await self._node_stop(cropper_client, f'Cropper {object_name}')
        await self._node_stop(yolo_client, f'YOLO {object_name}')

        # --- RESULT PACKAGING ---
        pose = self.vision_data[object_name]['pose']
        radius = self.vision_data[object_name]['radius']

        if pose is not None:
            result_msg.success = True
            result_msg.poses = [pose]
            x, y, z = pose.pose.position.x, pose.pose.position.y, pose.pose.position.z
            result_msg.message = f"Successfully detected {display_label} at [x: {x:.3f}, y: {y:.3f}, z: {z:.3f}]"
            if radius is not None:
                result_msg.message += f" | Radius: {radius:.4f}m"
        else:
            result_msg.success, result_msg.message = False, f"Vision timeout: {display_label} not found."

        return result_msg

    async def _set_remote_parameters(self, node_name, param_dict):
        """Sets parameters on a remote node asynchronously using the SetParameters service."""
        param_service = f"{node_name}/set_parameters"
        client = self.create_client(SetParameters, param_service, callback_group=self.group)
        
        if not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(f"Parameter service {param_service} not available.")
            return False

        request = SetParameters.Request()
        for name, value in param_dict.items():
            request.parameters.append(Parameter(name=name, value=value))
        
        future = client.call_async(request)
        while not future.done():
            await asyncio.sleep(0.05)
        
        return True

    async def handle_subdoor_pose(self, goal_handle):
        feedback_msg, result_msg = RunVision.Feedback(), RunVision.Result()
        self.vision_data['subdoor'] = None

        feedback_msg.current_phase = "Activating Subdoor Node..."
        goal_handle.publish_feedback(feedback_msg)
        if not await self._node_start(self.subdoor_lifecycle_client, 'Subdoor Pose Node'):
            result_msg.success, result_msg.message = False, "Failed to activate subdoor."
            return result_msg 

        feedback_msg.current_phase = "Waiting for subdoor..."
        goal_handle.publish_feedback(feedback_msg)

        for _ in range(150):
            if self.vision_data['subdoor'] is not None: break
            await asyncio.sleep(0.1)

        feedback_msg.current_phase = "Deactivating Subdoor Node..."
        goal_handle.publish_feedback(feedback_msg)
        await self._node_stop(self.subdoor_lifecycle_client, 'Subdoor Pose Node')

        markers = self.vision_data['subdoor']
        if markers is not None:
            result_msg.success = True
            pose_info = [f"Subdoor estimated successfully with {len(markers)} corner markers."]
            for i, marker in enumerate(markers):
                x, y, z = marker.pose.position.x, marker.pose.position.y, marker.pose.position.z
                pose_info.append(f"Pose {i}: x={x:.3f}, y={y:.3f}, z={z:.3f}")
            result_msg.message = "\n".join(pose_info)
        else:
            result_msg.success, result_msg.message = False, "Timed out waiting for subdoor."

        return result_msg

    async def handle_detect_screwdriver(self, goal_handle):
        feedback_msg, result_msg = RunVision.Feedback(), RunVision.Result()
        self.vision_data['screwdriver'] = None

        feedback_msg.current_phase = "Activating OBB Pipeline..."
        goal_handle.publish_feedback(feedback_msg)

        obb_success = await self._node_start(self.obb_object_lifecycle_client, 'OBB Detector Node')
        cropper_success = await self._node_start(self.obb_cropper_lifecycle_client, 'OBB Cropper Node')

        if not (obb_success and cropper_success):
            result_msg.success, result_msg.message = False, "Failed to activate OBB nodes."
            return result_msg

        feedback_msg.current_phase = "Waiting for screwdriver..."
        goal_handle.publish_feedback(feedback_msg)

        for _ in range(150):
            if self.vision_data['screwdriver'] is not None: break
            await asyncio.sleep(0.1)

        feedback_msg.current_phase = "Deactivating OBB Pipeline..."
        goal_handle.publish_feedback(feedback_msg)
        
        await self._node_stop(self.obb_cropper_lifecycle_client, 'OBB Cropper Node')
        await self._node_stop(self.obb_object_lifecycle_client, 'OBB Detector Node')

        pose = self.vision_data['screwdriver']
        if pose is not None:
            result_msg.success = True
            x, y, z = pose.pose.position.x, pose.pose.position.y, pose.pose.position.z
            qx, qy, qz, qw = pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w
            result_msg.message = (f"Screwdriver detected at [x: {x:.3f}, y: {y:.3f}, z: {z:.3f}] "
                                 f"Orientation: [qx: {qx:.3f}, qy: {qy:.3f}, qz: {qz:.3f}, qw: {qw:.3f}]")
        else:
            result_msg.success, result_msg.message = False, "Timed out waiting for screwdriver."

        return result_msg

    async def handle_place_object(self, goal_handle):
        feedback_msg, result_msg = RunVision.Feedback(), RunVision.Result()
        self.vision_data['place_pose'] = None

        # --- DYNAMIC PARAMETER OVERRIDE (radius) ---
        radius = goal_handle.request.radius
        if radius > 0.0:
            self.get_logger().info(f"Setting place_object radius to: {radius:.4f}m")
            await self._set_remote_parameters('/place_object_node', {
                'test_radius': ParameterValue(
                    type=ParameterType.PARAMETER_DOUBLE,
                    double_value=float(radius)
                )
            })
        else:
            self.get_logger().info("No radius provided, using default test_radius on place_object_node.")

        feedback_msg.current_phase = "Activating Table Segmentation (Place) Node..."
        goal_handle.publish_feedback(feedback_msg)

        if not await self._node_start(self.place_object_client, 'Place Object Node'):
            result_msg.success, result_msg.message = False, "Failed to activate Place Object node."
            return result_msg

        feedback_msg.current_phase = "Scanning table for safe dropping zone..."
        goal_handle.publish_feedback(feedback_msg)

        for _ in range(150):
            if self.vision_data['place_pose'] is not None: break
            await asyncio.sleep(0.1)

        feedback_msg.current_phase = "Deactivating Place Object Node..."
        goal_handle.publish_feedback(feedback_msg)
        await self._node_stop(self.place_object_client, 'Place Object Node')

        pose = self.vision_data['place_pose']
        if pose is not None:
            result_msg.success = True
            x, y, z = pose.pose.position.x, pose.pose.position.y, pose.pose.position.z
            qx, qy, qz, qw = pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w
            result_msg.message = (f"Safe Placement Pose found at [x: {x:.3f}, y: {y:.3f}, z: {z:.3f}]\n"
                                f"Orientation: [qx: {qx:.3f}, qy: {qy:.3f}, qz: {qz:.3f}, qw: {qw:.3f}]")
            if radius > 0.0:
                result_msg.message += f"\nRadius used: {radius:.4f}m"
        else:
            result_msg.success, result_msg.message = False, "Timed out waiting for an empty spot on the table."

        return result_msg

    # --- SHUTDOWN ROUTINES ---
    def sync_shutdown_routine(self):
        self.get_logger().info('Initiating graceful shutdown of all pipeline nodes...')
        managed_nodes = [
            ('YOLO Screw Node',    self.yolo_screw_client),
            ('YOLO Car Node',      self.yolo_car_client),
            ('Cropper Screw Node', self.cropper_screw_client),
            ('Cropper Car Node',   self.cropper_car_client),
            ('Table Height Node',  self.table_lifecycle_client),
            ('Subdoor Pose Node',  self.subdoor_lifecycle_client),
            ('OBB Detector Node',  self.obb_object_lifecycle_client),
            ('OBB Cropper Node',   self.obb_cropper_lifecycle_client),
            ('Place Object Node',  self.place_object_client),
        ]

        for name, client in managed_nodes:
            if not self._lazy:
                # Eager mode: nodes are in Configured state at rest, need cleanup first
                self.get_logger().info(f'  Cleaning up {name}...')
                self._sync_change_state(client, Transition.TRANSITION_CLEANUP)
            # In lazy mode: nodes are already Unconfigured at rest, skip cleanup
            self.get_logger().info(f'  Shutting down {name}...')
            self._sync_change_state(client, Transition.TRANSITION_UNCONFIGURED_SHUTDOWN)

        self.get_logger().info('All managed nodes shut down cleanly.')

    def _sync_change_state(self, client, transition_id):
        if not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(f"Service {client.srv_name} not available during shutdown.")
            return False
            
        req = ChangeState.Request()
        req.transition.id = transition_id
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        
        if future.result() is not None:
            return future.result().success
        return False

def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    
    node = PerceptionDispatcher()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl+C caught! Beginning graceful shutdown...")
    finally:
        node.sync_shutdown_routine()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()