This repository contains the code for our work **[Topologically Densified Distributions](https://arxiv.org/abs/2002.04805)** which was presented at ICML'20.


# Installation

In the following `<root_dir>` will be the directory you have chosen for the installation.

1. Install Anaconda from [here](https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh) into `<root_dir>/anaconda3`, i.e., set the prefix accordingly in the installer. Do **not** initialize your shell via the installer (the installer asks you this at the end of the installation). 

2. Activate Anaconda installation: 

    ```
    eval "$(<root_dir>/anaconda3/bin/conda shell.bash hook)"

    ```


3. Install pytorch via conda

    ```
    conda install torchvision cudatoolkit=<your_cuda_version> -c pytorch
    ```

4. Install other dependencies

    ```
    pip install fastprogress
    ```


5. Install `torchph` via 

    ```
    cd <root_dir>
    git clone -b 'submission_icml2020' --single-branch --depth 1 https://github.com/c-hofer/torchph.git
    conda develop torchph
    ```
6. Clone this repository into `<root_dir>`. 

# Application

1. Use the `run_experiments.py` script to run experiments. Pre-configured is an experiment on `cifar10` with the proposed regularization. Alter the script to run different experiments (see `run_experiments.py`). 
If you run the script, each experiment gets a unique id and its output is written into a sub-folder for the `results` directory. 

2. The notebook `explore_results.ipynb` contains some code to browse the results. 




