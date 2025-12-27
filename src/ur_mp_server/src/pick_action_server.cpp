// pick_action_server.cpp

#include <memory>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "ur_package_msgs/action/pick_action.hpp"

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_interface/planning_interface.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/planning_scene/planning_scene.h>

int main(int argc, char **argv)
{
  // 1) Initialize ROS 2
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("pick_action_server");

  // 2) Create MoveGroupInterface for "manipulator"
  auto move_group = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
      node, "manipulator");
  move_group->setPlanningTime(10.0);
  RCLCPP_INFO(node->get_logger(), "MoveGroup for 'manipulator' ready");

  // 3) Action type aliases
  using Pick = ur_package_msgs::action::PickAction;
  using GoalHandle = rclcpp_action::ServerGoalHandle<Pick>;

  // 4) Create the action server
  auto action_server = rclcpp_action::create_server<Pick>(
      node,
      "pick_action",
      // --- Goal callback ---
      [&](const rclcpp_action::GoalUUID &,
          std::shared_ptr<const Pick::Goal>)
      {
        RCLCPP_INFO(node->get_logger(), "Goal received: ACCEPT");
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
      },
      // --- Cancel callback ---
      [&](const std::shared_ptr<GoalHandle>)
      {
        RCLCPP_INFO(node->get_logger(), "Cancel request: ACCEPT");
        return rclcpp_action::CancelResponse::ACCEPT;
      },
      // --- Accept callback ---
      [&](const std::shared_ptr<GoalHandle> goal_handle)
      {
        std::thread{
            [node, move_group, goal_handle]()
            {
              RCLCPP_INFO(node->get_logger(), "Executing pick action...");

              // 5) Extract the target pose from goal
              auto target = goal_handle->get_goal()->source;

              // 6) Plan to the target
              move_group->setPoseTarget(target);
              moveit::planning_interface::MoveGroupInterface::Plan plan;

              bool ok = (move_group->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);

              if (!ok)
              {
                RCLCPP_WARN(node->get_logger(), "Planning failed");
              }
              else
              {
                // 7) Execute the plan
                ok = (move_group->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
                move_group->stop();
                move_group->clearPoseTargets();

                if (!ok)
                {
                  RCLCPP_WARN(node->get_logger(), "Execution failed");
                }
              }

              // 8) Send result
              auto result = std::make_shared<Pick::Result>();
              result->result = ok;

              if (ok)
              {
                goal_handle->succeed(result);
                RCLCPP_INFO(node->get_logger(), "PickAction succeeded");
              }
              else
              {
                goal_handle->abort(result);
                RCLCPP_ERROR(node->get_logger(), "PickAction aborted");
              }
            }}
            .detach();
      });

  RCLCPP_INFO(node->get_logger(), "PickActionServer ready");
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
