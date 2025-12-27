import math
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg
from isaaclab_assets import UR10_CFG  # Reuse base config
from copy import deepcopy


@configclass
class UR16eReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Set correct robot USD prim
        #self.scene.robot = UR10_CFG.replace(prim_path="/World/ur5")
        ur16e_cfg = deepcopy(UR10_CFG)
        ur16e_cfg.prim_path = "/World/ur5"  # Articulation root in the stage
        ur16e_cfg.usd_path = "/home/alien/workspace/ros_ur_driver/src/Universal_Robots_ROS2_Driver/ur_description/urdf/ur16e/ur16e.usd"  # File path to USD

        self.scene.robot = ur16e_cfg

        # Reset joint range
        self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)

        # Use correct EE link name
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["ee_link"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["ee_link"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["ee_link"]

        # Control all joints
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=0.5,
            use_default_offset=True,
        )

        # Command targets the EE link
        self.commands.ee_pose.body_name = "ee_link"
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)


@configclass
class UR16eReachEnvCfg_PLAY(UR16eReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
