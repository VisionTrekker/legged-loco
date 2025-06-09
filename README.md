# Legged Loco
This repo is used to train low-level locomotion policy of Unitree Go2 and H1 in Isaac Lab.

<p align="center">
<img src="./src/go2_teaser.gif" alt="First Demo" width="45%">
&emsp;
<img src="./src/h1_teaser.gif" alt="Second Demo" width="45%">
</p>


## Installation
1. Create a new conda environment with python 3.10.
    ```shell
    conda create -n isaaclab python=3.10
    conda activate isaaclab
    ```

2. Install PyTorch with CUDA 121.
    ```shell
    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
    ```
   For 50 series GPU, you could install PyTorch 2.7.0 with CUDA 128.
    ```shell
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    ``` 

3. To update pip.
    ```shell
    pip install --upgrade pip
    ```

4. Make sure that Isaac Sim is installed on your machine. Otherwise follow [this guideline](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) to install it. Now it is suit for Isaac Sim 4.5.0 + IsaacLab 2.1.0. On Ubuntu 22.04 or higher, you could install it via pip:
    ```shell
    pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
    ```

5. Clone the Isaac Lab 2.1.0 repository and run the Isaac Lab installer script.
    ```shell
    git clone git@github.com:isaac-sim/IsaacLab.git
    cd IsaacLab
    # Install dependencies using apt (on Ubuntu).
    sudo apt install cmake build-essential
    # install all the learning frameworks.
    ./isaaclab.sh --install # or "./isaaclab.sh -i"
    ```

5. Additionally install rsl rl and tasks in this repo.
    ```shell
    python -m pip install -e {THIS_REPO_DIR}/source/leggedloco_rl
    python -m pip install -e {THIS_REPO_DIR}/source/leggedloco_tasks
    ```


## Usage
### Available Environments
*
    ```shell
    python scripts/list_envs.py
    ```


### Train
#### Flat terrain
*
    ```shell
    # Blind
    python scripts/train.py --task LeggedLoco-AlienGo-Flat --history_len 9 --run_name blind --max_iterations 2000 --save_interval 200 --headless
    # With 360 lidar
    python scripts/train.py --task LeggedLoco-AlienGo-Flat-Lidar --history_len 9 --run_name lidar --max_iterations 2000 --save_interval 200 --headless
    ```

#### Rough (stairs) terrain
* Scratch
    ```shell
    # Blind
    python scripts/train.py --task LeggedLoco-AlienGo-Rough --history_len 9 --run_name blind --max_iterations 3000 --save_interval 200 --headless
    # With 360 lidar
    python scripts/train.py --task LeggedLoco-AlienGo-Rough-Lidar --history_len 9 --run_name lidar --max_iterations 3000 --save_interval 200 --headless
    ```
* Two stage
    ```shell
    # Blind
    python scripts/train.py --task LeggedLoco-AlienGo-Rough --history_len 9 --run_name blind_loadflat --resume True --load_experiment aliengo_flat --load_run "2025-06-04_14-00-01_blind_jump" --max_iterations 2600 --save_interval 200 --headless
    # With 360 lidar
    python scripts/train.py --task LeggedLoco-AlienGo-Rough-Lidar --history_len 9 --run_name lidar_loadflat --resume True --load_experiment aliengo_flat --load_run "2025-06-04_11-42-50_lidar" --max_iterations 2600 --save_interval 200 --headless
    ```


### Test
* 
    ```shell
    python scripts/play.py --task TASK_ID-Play --history_len 9 --load_run RUN_NAME
    ```
*   For Saving video
    ```shell
    python scripts/play.py --task TASK_ID-Play --history_len 9 --load_run RUN_NAME --headless --video
    ```

## Add New Environments
You can add additional environments by placing them under `/source/leggedloco_tasks/leggedloco_tasks/manager_based/locomotion/config`.