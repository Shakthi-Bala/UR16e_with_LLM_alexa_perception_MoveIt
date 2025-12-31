# RL-Based Reach Control for UR16e (Isaac Sim ↔ ROS 2 Sim2Real) 🤖🦾

This project implements a **Reinforcement Learning (PPO)-based reach policy** for a **UR16e robotic arm**, trained in **Isaac Sim 5.1.0** and deployed using **ROS 2** for both **simulation** and **real hardware execution**.

The repository contains:
- RL training and evaluation pipelines (IsaacLab + RSL-RL)
- Simulation assets and logs
- ROS 2 + MoveIt integration
- Sim-to-real deployment workflow

---

## 🧩 Repository Overview

```bash
.
├── Work_for_paper/                     # Experimental data & paper-related material
├── isaaclab_ur_reach_sim2real/         # IsaacLab RL task & configs
├── logs/rsl_rl/                        # Training logs & checkpoints
├── outputs/                            # Evaluation outputs
├── scripts/                            # Training & evaluation scripts
├── source/                             # IsaacLab source extensions
├── src/                                # ROS 2 / auxiliary source
├── README.md
├── *.webm                              # Demo videos (simulation & real robot)
├── *.usd                               # Isaac Sim USD assets
```
## 🧠 System Overview

- **Simulator:** Isaac Sim 5.1.0
- **RL Algorithm:** PPO (via RSL-RL)
- **Robot:** Universal Robots UR16e
- **Middleware:** ROS 2
- **Motion Planning (baseline):** MoveIt

### 🚀 Deployment Modes

- **Isaac Sim**
  - Headless mode
  - GUI mode
- **ROS 2 Fake Hardware**
- **ROS 2 Real Hardware (UR16e)**

## 🧰 Requirements
### Software:
- Ubuntu 20.04 / 22.04
- Isaac Sim 5.1.0
- IsaacLab
- ROS 2 (Humble / Iron recommended)
- Python 3.8+
- Universal Robots ROS 2 Driver
- MoveIt 2
⚠️ Important DDS Note
Do NOT use Cyclone DDS.
Simulation and hardware execution will not work with Cyclone DDS.

## 🧪 RL Training & Evaluation (Isaac Sim)
### ▶️ Train the RL Agent
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Reach-UR10-v0 \
  --headless
```
### ▶️ Generate Policy Checkpoint (.pt)
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Reach-UR10-v0 \
  --headless
```

### ▶️ Evaluate / Test a Trained Policy
```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Reach-UR16e-v0 \
  --checkpoint /home/alien/IsaacLab/logs/rsl_rl/reach_ur10/2025-07-23_10-16-55/model_999.pt \
  --video \
  --num_envs 1 \
  --rendering_mode quality \
  --real-time
```
This command:
- Loads a trained .pt model
- Runs a single environment
- Records a video
- Uses real-time rendering
  
## 🤖 Running the Robot on Real Hardware
### Network Configuration
```bash
robot_ip     : 192.168.1.6
pc_static_ip : 192.168.1.7
```
### Terminal 1 – UR Driver
```bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur16e \
  robot_ip:=192.168.1.6 \
  kinematics_params_file:=/home/alien/my_robot_calibration.yaml
```
### Terminal 2 – MoveIt
```bash
ros2 launch ur_moveit_config ur_moveit.launch.py \
  ur_type:=ur16e \
  launch_rviz:=true
```

### Terminal 3 – Activate Controller
```bash
ros2 control switch_controllers \
  --activate scaled_joint_trajectory_controller \
  --strict
```
### View TF Frames
```bash
ros2 run tf2_tools view_frames
```

## 🧪 Running the Robot in Software (Fake Hardware)
### Terminal 1 – Fake Hardware Driver
```bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur16e \
  robot_ip:=192.168.1.6 \
  use_fake_hardware:=true
```
### Terminal 2 – MoveIt
```bash
ros2 launch ur_moveit_config ur_moveit.launch.py \
  ur_type:=ur16e \
  launch_rviz:=true
```

### Terminal 3 – Activate Controller
```bash
ros2 control switch_controllers \
  --activate joint_trajectory_controller \
  --strict
```
### Terminal 4 – Test with Custom XYZ Goal
```bash
ros2 run ur_moveit_cpp_ext moveit_xyz
```

## 🧰 Useful Utilities
### Convert Xacro → URDF
```bash
xacro ur.urdf.xacro \
  ur_type:=ur16e \
  name:=ur16e \
  prefix:="" \
  robot_ip:=192.168.1.10 \
  force_abs_paths:=false > ur16e.urdf
```
### Check Package Install Location
```bash
ros2 pkg prefix <package_name>
```

If not found:
```bash
source install/setup.bash
```

## 🖥️ Saving RViz Configuration
### Terminal 1
```bash
ros2 launch ur_description view_ur.launch.py ur_type:=ur16e
```
### Terminal 2
```bash
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map base_link
```
### Terminal 3
```bash
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map world
```
Then save the RViz configuration (.rviz) from the GUI.

## 📜 Notes & Limitations
- PPO policy trained in Isaac Sim 5.1.0
- MoveIt used as a baseline comparison
- Sim-to-real gap handled via calibrated URDF & kinematics
- DDS configuration is critical (avoid Cyclone DDS)
- Safety checks required before real robot execution

## 🎥 Demos
Included .webm files demonstrate:
- RL reach in simulation
- RL reach in Isaac Sim
- MoveIt-based reach 
- Real robot execution

## 👤 Author
Shakthi Bala
Reinforcement Learning | Robotic Manipulation | Isaac Sim | ROS 2 | Sim2Real
---

If you want next, I can:
- Add a **Sim2Real architecture diagram**
- Convert this into a **paper-companion README**
- Add **baseline comparison table (MoveIt vs PPO)**
- Polish this into a **flagship portfolio project**

Just say 👍
