# file: shared_ros2_manager.py

import threading
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

class SharedROS2Manager:
    """
    A singleton-like manager to handle a single, shared rclpy executor
    for multiple robot objects in the same process.
    """
    _executor: MultiThreadedExecutor | None = None
    _executor_thread: threading.Thread | None = None
    _lock = threading.Lock()
    _active_nodes: set[Node] = set()

    @classmethod
    def add_node(cls, node: Node):
        """Adds a node to the shared executor, starting it if it's the first."""
        with cls._lock:
            if not rclpy.ok():
                rclpy.init()

            if cls._executor is None:
                print("--- Initializing shared ROS2 executor ---")
                cls._executor = MultiThreadedExecutor()
                cls._executor_thread = threading.Thread(target=cls._executor.spin, daemon=True)
                cls._executor_thread.start()

            print(f"--- Adding node '{node.get_name()}' to shared executor. ---")
            cls._executor.add_node(node)
            cls._active_nodes.add(node)

    @classmethod
    def ensure_initialized(cls):
        """
        Ensures that rclpy.init() has been called. This should be called
        before any rclpy objects (like a Node) are created.
        """
        with cls._lock:
            if not rclpy.ok():
                print("--- Initializing rclpy context ---")
                rclpy.init()
    @classmethod
    def remove_node(cls, node: Node):
        """Removes a node and shuts down the executor if it's the last one."""
        with cls._lock:
            if node not in cls._active_nodes:
                return # Node already removed or was never added

            if cls._executor:
                print(f"--- Removing node '{node.get_name()}' from shared executor. ---")
                cls._executor.remove_node(node)
                
            cls._active_nodes.remove(node)
            print(f"--- Node removed. Active nodes remaining: {len(cls._active_nodes)} ---")

            if not cls._active_nodes and cls._executor:
                print("--- Last node removed. Shutting down shared ROS2 executor. ---")
                cls._executor.shutdown()
                if cls._executor_thread and cls._executor_thread.is_alive():
                    cls._executor_thread.join(timeout=1.0) # Add a timeout
                cls._executor = None
                cls._executor_thread = None
                # Do NOT call rclpy.shutdown() here, let the main process handle it.