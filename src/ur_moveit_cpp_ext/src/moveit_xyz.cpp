#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometry_msgs/msg/pose.hpp>

#include <memory>

void move_robot(const std::shared_ptr<rclcpp::Node> node)
{
    // Create MoveGroupInterface object for the arm
    auto arm_move_group = moveit::planning_interface::MoveGroupInterface(node, "ur_manipulator");

    // ---------- Arm Planning to XYZ Pose ----------
    // Define the desired pose for the end-effector
    geometry_msgs::msg::Pose target_pose;
    target_pose.position.x = 0.4;  // Replace with your target x-0.385
    target_pose.position.y =  0.0  ;  // Replace with your target y-0.186 
    target_pose.position.z = 0.3;  // Replace with your target z1.076

    // Set the orientation (unit quaternion - no rotation)
    target_pose.orientation.x = 0.0;
    target_pose.orientation.y = 0.0;
    target_pose.orientation.z = 0.0;
    target_pose.orientation.w = 1.0;

    // Set the pose target
    arm_move_group.setPoseTarget(target_pose);

    // Plan to the pose target
    moveit::planning_interface::MoveGroupInterface::Plan arm_plan;
    bool arm_plan_success = (arm_move_group.plan(arm_plan) == moveit::core::MoveItErrorCode::SUCCESS);

    // ---------- Execute Plan if Successful ----------
    if (arm_plan_success)
    {
        RCLCPP_INFO(node->get_logger(), "Arm planning succeeded. Executing motion...");
        arm_move_group.move();
    }
    else
    {
        RCLCPP_ERROR(node->get_logger(), "Arm planning failed.");
    }
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    auto node = rclcpp::Node::make_shared("simple_moveit_interface");
    move_robot(node);

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}