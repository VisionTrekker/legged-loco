# Legged Loco
This repo is used to train low-level locomotion policy of Unitree Go2 and H1 in Isaac Lab.

<p align="center">
<img src="./src/go2_teaser.gif" alt="First Demo" width="45%">
&emsp;
<img src="./src/h1_teaser.gif" alt="Second Demo" width="45%">
</p>


## Installation

1. Create a new conda environment with python 3.11. (Use isaaclab 2.2 + isaacsim 5.0 needs 3.11)
    ```shell
    conda create -n isaaclab python=3.11
    conda activate isaaclab
    ```

2. Install PyTorch with CUDA 128.
    ```shell
    pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
    ```

3. To update pip.
    ```shell
    pip install --upgrade pip
    ```

4. Install the Isaac Sim5.0 [this guideline](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) to 
   install it (suit for Isaac Sim 5.0 + IsaacLab 2.2.0). On Ubuntu 22.04 or higher, you could install it via pip:
    ```shell
    pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com --timeout=600 --retries=5 --resume-retries=5
    ```
   or from source follow Isaac Sim [github guideline](https://github.com/isaac-sim/IsaacSim.git)

5. Clone the Isaac Lab 2.2.0 repository and run the Isaac Lab installer script.
    ```shell
    git clone git@github.com:isaac-sim/IsaacLab.git
    cd IsaacLab
    # Install dependencies using apt (on Ubuntu).
    sudo apt install cmake build-essential
    # install all the learning frameworks.
    ./isaaclab.sh --install # or "./isaaclab.sh -i"
    ```
    if Isaac Sim is built from source, the IsaacLab installation is as follows:
    ```shell
    git clone https://github.com/isaac-sim/IsaacLab.git
    cd IsaacLab
    ln -s ../isaacsim/_build/linux-x86_64/release _isaac_sim
    ./isaaclab.sh --install
    # setup for conda environment (everytime when open new termination!)
    source _isaac_sim/setup_conda_env.sh
    ```

5. Additionally install tasks in this repo.
    ```shell
    python -m pip install -e {THIS_REPO_DIR}/source/leggedloco_tasks
    ```


## Training Reflections

| **大项** | **细则**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
|----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **两阶段训练策略** | 直接在多地形下训练会导致 AlienGo 步态不稳定。<br> 1. 第一阶段：在平地（光滑 + 粗糙）训练稳定行走步态<br>2. 第二阶段：在多地形（平地 + 斜坡 + 楼梯）训练上下楼梯能力                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| **地形配比** | - 第一阶段：50% 光滑平地 + 50% 粗糙平地<br>- 第二阶段：10% 粗糙平地 + 10% 下光滑斜坡 + 10% 上光滑斜坡 + 30% 下楼梯 + 40% 上楼梯<br><br>**说明**：针对巡检场景，增加上下楼梯比例，并保留少量平地以维持基本步态。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| **步态调整** | 1. **摩擦力域随机化**<br>&nbsp;&nbsp;&nbsp;&nbsp;- 平地训练：静摩擦力 `(0.4, 2.0)`，动摩擦力 `(0.4, 1.2)`，避免四足滑动导致步态退化为前后跳跃（LiDAR 模式设置为静摩擦力 `(0.4, 2.0)`，动摩擦力 `(0.4, 1.6)`）<br>&nbsp;&nbsp;&nbsp;&nbsp;- 多地形训练：静摩擦力 `(0.6, 0.9)`，动摩擦力 `(0.6, 0.8)`，模拟瓷砖楼梯低摩擦环境 (0.2摩擦力太小，机器狗会滑步前进)<br><br>2. **四足奖励函数优化**<br>&nbsp;&nbsp;&nbsp;&nbsp;- 问题：原`feet_air_time` 仅鼓励离地时间 > 0.5s，缺乏协调性激励<br>&nbsp;&nbsp;&nbsp;&nbsp;- 改进：同时考虑离地时间和接触时间，适应不同速度需求（高速优化步频，低速优化稳定性）<br><br>3. **关节接触惩罚**<br>&nbsp;&nbsp;&nbsp;&nbsp;- 平地：惩罚 `calf` 关节接触，避免肘关节离地太近<br>&nbsp;&nbsp;&nbsp;&nbsp;- 多地形：惩罚 `hip`、`thigh`、`calf` 全关节接触，避免触碰楼梯<br><br>4. **基座高度惩罚**<br>&nbsp;&nbsp;&nbsp;&nbsp;- `asset.data.root_pos_w[:, 2]` 获取是 `base` 相对 `ground` 的高度<br>&nbsp;&nbsp;&nbsp;&nbsp;- rough 地形不准确，因此仅在平地训练中添加惩罚，多地形未使用 |
| **基座碰撞节点** | - **问题**：AlienGo 的 USD 文件中具有碰撞检测的是 `"trunk"` 而非 `"base"`<br>- **解决方案**：在触发终止、域随机化推力等操作中，将基座 `body_names` 改为 `"trunk"`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| **执行器类型** | - **配置**：虽然 AlienGo 关节为直流电机（`DCMotor`），但为应对实际中的延迟，选用 `DelayedPDActuator` 模拟延迟<br>- **延时范围**：`(4, 6)` 个控制周期（即 `(4*0.005, 6*0.005)` 秒）                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| **重置初始方向** | - **配置**：<br>&nbsp;&nbsp;&nbsp;&nbsp;- reset时 yaw 角度范围：`(-0.157, 0.157)`（±9°）<br>&nbsp;&nbsp;&nbsp;&nbsp;- commands 中 yaw 角度范围：`(-0.1, 0.1)`（±5.7°）<br>- **目的**：模拟非正对楼梯的运动场景，提升上下楼梯的鲁棒性                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
* 在 IsaacLab 中，Aliengo 初始位置为 (-0.1, 0.1, 0.8, 1.0, -1.5) 比 (-0.0, 0.0, 0.8, 0.8, -1.5) 的训练收敛快。
* history_len 由 9 改为 5，且新的观测帧放在前面，训练上限更高。

## Usage

### Available Environments
*
    ```shell
    python scripts/tools/list_envs.py
    ```

    | **Environment ID**                                       | **Description**                                                                                                     |
    |----------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
    | LeggedLoco-Isaac-Velocity-Flat-AlienGo-v0                | Track a velocity command on flat terrain with the Unitree AlienGo robot (blind walking)                             |
    | LeggedLoco-Isaac-Velocity-Flat-Play-AlienGo-v            |                                                                                                                     |
    | LeggedLoco-Isaac-Velocity-Flat-Lidar-AlienGo-v0          | Track a velocity command on flat terrain with the Unitree AlienGo robot enhanced by Mid-360 LiDAR                   |
    | LeggedLoco-Isaac-Velocity-Flat-Lidar-Play-AlienGo-v0     |                                                                                                                     |
    | LeggedLoco-Isaac-Velocity-Rough-AlienGo-v0               | Track a velocity command on rough terrain (slope + stairs) with the Unitree AlienGo robot (blind walking)           |
    | LeggedLoco-Isaac-Velocity-Rough-Play-AlienGo-v0          |                                                                                                                     |
    | LeggedLoco-Isaac-Velocity-Rough-Lidar-AlienGo-v0         | Track a velocity command on rough terrain (slope + stairs) with the Unitree AlienGo robot enhanced by Mid-360 LiDAR |
    | LeggedLoco-Isaac-Velocity-Rough-Lidar-Play-AlienGo-v0    |                                                                                                                     |
    | LeggedLoco-Isaac-Velocity-Flat-Recover-AlienGo-v0        | Track a velocity command on flat terrain with the Unitree AlienGo robot (blind walking + recover)                   |
    | LeggedLoco-Isaac-Velocity-Flat-Recover-Play-AlienGo-v0   |                                                                                                                     |
    | LeggedLoco-Isaac-Velocity-Rough-Recover-AlienGo-v0       | Track a velocity command on rough terrain (slope + stairs) with the Unitree AlienGo robot (blind walking + recover) |
    | LeggedLoco-Isaac-Velocity-Rough-Recover-Play-AlienGo-v0  |                                                                                                                     |


### Train
#### Flat terrain
*
    ```shell
    # Blind
    python scripts/rsl_rl/base/train.py --task LeggedLoco-Isaac-Velocity-Flat-AlienGo-v0 --history_len 5 --run_name blind --max_iterations 2000 --save_interval 200 --headless
    # Blind + Recover
    python scripts/rsl_rl/base/train.py --task LeggedLoco-Isaac-Velocity-Flat-Recover-AlienGo-v0 --history_len 5 --run_name blind_load --resume --load_experiment aliengo_flat --load_run "2025-06-27_16-17-06_blind_good" --max_iterations 2500 --save_interval 200 --headless
    
    # Vision (lidar mid-360) (not verified after env_cfg updated)
    python scripts/rsl_rl/base/train.py --task LeggedLoco-Isaac-Velocity-Flat-Lidar-AlienGo-v0  --history_len 5 --run_name lidar --max_iterations 2000 --save_interval 200 --headless
    ```

#### Rough (stairs) terrain
* Two stages
    ```shell
    # Blind + Recover
    python scripts/rsl_rl/base/train.py --task LeggedLoco-Isaac-Velocity-Rough-Recover-AlienGo-v0 --history_len 5 --run_name blind_loadflat --resume --load_experiment aliengo_flatRecover --load_run "2025-07-01_15-36-26_blind_load" --max_iterations 2500 --save_interval 200 --headless
    
    # Vision (lidar mid-360) (not verified after env_cfg updated)
    python scripts/rsl_rl/base/train.py --task LeggedLoco-Isaac-Velocity-Rough-Lidar-AlienGo-v0 --history_len 5 --run_name lidar_loadflat --resume --load_experiment aliengo_flat --load_run "2025-06-10_18-32-28_lidar" --max_iterations 2500 --save_interval 200 --headless
    ```

* Scratch
    ```shell
    # Blind (not verified after env_cfg updated)
    python scripts/rsl_rl/base/train.py --task LeggedLoco-Isaac-Velocity-Rough-AlienGo-v0 --history_len 5 --run_name blind --max_iterations 3000 --save_interval 200 --headless
    
    # Vision (lidar mid-360) (not verified after env_cfg updated)
    python scripts/rsl_rl/base/train.py --task LeggedLoco-Isaac-Velocity-Rough-Lidar-AlienGo-v0 --history_len 5 --run_name lidar --max_iterations 3000 --save_interval 200 --headless
    ```


### Test
* 
    ```shell
    python scripts/rsl_rl/base/play.py --task TASK_ID-Play --history_len 5 --load_run RUN_NAME
    ```
*   For Saving video
    ```shell
    python scripts/rsl_rl/base/play.py --task TASK_ID-Play --history_len 5 --load_run RUN_NAME --headless --video
    ```

## Add New Environments

### convert urdf to usd
    ```shell
    python scripts/tools/convert_urdf.py --input <INPUT_URDF> --output <OUTPUT_USD>
    ```
> [!NOTE]
> There is a problem with the urdf conversion of the current Isaacsim version. Please checkout your IsaacLab to **v1.4.1**.


### configure new env
You can add additional environments by placing them under `/source/leggedloco_tasks/leggedloco_tasks/manager_based/locomotion/velocity/config`.

```tree
source/leggedloco_tasks/leggedloco_tasks/manager_based/locomotion/velocity
├── __init__.py
└── velocity
    ├── config
    │   ├── humanoid
    │   └── quadruped
    │       └── unitree_a1
    │           ├── agent  # <- this is where we store the learning agent configurations
    │           ├── __init__.py  # <- this is where we register the environment and configurations to gym registry
    │           ├── flat_env_cfg.py
    │           └── rough_env_cfg.py
    ├── __init__.py
    ├── mdp
    ├── utils
    │   └── wrappers.py
    ├── velocity_env_cfg.py  # <- this is the base task configuration
    └── velocity_recover_env_cfg.py
```