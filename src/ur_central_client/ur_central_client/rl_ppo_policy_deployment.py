import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import torch

class PolicyControllerNode(Node):
    def __init__(self):
        super().__init__('policy_controller')
        # Load the trained policy model
        self.policy = torch.jit.load('reach_policy.pt')
        self.policy.eval()
        self.get_logger().info("Policy model loaded.")
        # Subscribe to joint states from the robot
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        # Publisher for joint command (using a forward position controller topic)
        self.cmd_pub = self.create_publisher(Float64MultiArray, '/forward_position_controller/commands', 10)
        # Timer to run control loop at 50 Hz (20ms interval)
        self.timer = self.create_timer(0.02, self.control_loop)
        self.current_state = None
    
    def joint_state_callback(self, msg: JointState):
        # Save the latest joint angles/velocities
        self.current_state = msg

    def control_loop(self):
        if self.current_state is None:
            return  # No state received yet
        # Construct observation vector from current_state (e.g., joint positions, velocities, target pose)
        obs = build_observation(self.current_state)  # user-defined function to form the RL observation
        # Run the policy network to get action
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        action = self.policy(obs_tensor)
        # Convert action to command message (e.g., target joint positions)
        cmd = Float64MultiArray()
        cmd.data = action.detach().cpu().numpy().tolist()
        self.cmd_pub.publish(cmd)
