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
    ./isaaclab.sh -p -m pip install -e {THIS_REPO_DIR}/source/leggedloco_rl
    ./isaaclab.sh -p -m pip install -e {THIS_REPO_DIR}/source/leggedloco_tasks
    
    # or
    cd {THIS_REPO_DIR}/source/leggedloco_rl
    pip install -e .
    cd ../leggedloco_tasks
    pip install -e .
    ```


## Usage
### train
* base (flat terrain)
    ```shell
    # height_scan
    python scripts/train.py --task=aliengo_base --history_len=9 --run_name=flat_heightscan --max_iterations=2000 --save_interval=200 --headless
    
    # 360 lidar
    python scripts/train.py --task=aliengo_base_lidar --history_len=9 --run_name=flat_lidar --max_iterations=2000 --save_interval=200 --headless
    ```
* vision (stairs terrain)
    ```shell
    # scratch
    python scripts/train.py --task=aliengo_vision --history_len=9 --run_name=stairs --max_iterations=10000 --save_interval=200 --headless
    
    # two stage
    python scripts/train.py --task=aliengo_vision --history_len=9 --run_name=stairs_loadflat --resume True --load_experiment aliengo_base --load_run="2025-06-04_11-42-50_flat_lidar" --max_iterations=5000 --save_interval=200 --headless
    ```

### test
* 
    ```shell
    python scripts/play.py --task=aliengo_base_play --history_len=9 --load_run=RUN_NAME
    ```

    Use `--headless` to enable headless mode. Add `--video` for headless rendering and video saving.

## Add New Environments
You can add additional environments by placing them under `/source/leggedloco_tasks/leggedloco_tasks/manager_based/locomotion/config`.