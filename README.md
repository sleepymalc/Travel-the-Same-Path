# README

## Setup

### Install Anaconda

We use `anaconda` to manage our environment. For people using Greatlakes, see the following. 

Since we can't use `sudo`, hence we build `anaconda` from source. Download [`Anaconda3-2021.11-Linux-x86_64.sh`](https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh) and upload it to your Greatlakes server using either [gui for greatlakes](https://greatlakes.arc-ts.umich.edu/pun/sys/dashboard) or [globus](https://www.globus.org/). After doing so, just run 

```bash
sh Anaconda3-2021.11-Linux-x86_64.sh
```

in the folder you put this script in.

### Create Virtual Environment

We'll need `ecole`, `pytorch` and also `pytorch_geometric`. Please follow the following configuration to reproduce the result. 

1. Create a environment

   ```bash
   conda create -n ecole python=3.8
   ```

2. Activate the newly created environment

   ```bash
   conda activate ecole
   ```

3. Install `ecole`

   ```bash
   conda conda install -c conda-forge ecole
   ```

4. Install `pytorch` with propriate CUDA version.

   ```bash
   pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
   ```

5. Install `pytorch_geometric` **without** `torch-spline-conv`. This is a buggy dependency and pyg(pytorch geometric) group decide to give up eventually as well. We simply ignore it since we don't need it anyway. 

   ```bash
   pip install torch-scatter torch-sparse torch-cluster torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
   ```

Notice that with `conda` install, `pip` will be installed as well by default and if one indeed activates the `conda` environment, using `pip` works as well. (packages are still installed in the virtual environment!)

To check whether your environment is working correctly, we test it as follows.

```bash
python -c 'import torch; print(torch.cuda.is_available())'
```

You should see `True` if you are in interactive mode with gpu resource. If you don't, then you'll probably see `False`, whcih is also fine. Just make sure you have test this at least once in interactive mode with gpu resource, which can be done by 

```bash
salloc --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem-per-cpu=16gb --account=<account> --time=1-00:00:00
```

for example.

## Running the Experiment

### `01_generate_instances.py`



### `02_generate_dataset.py`



After running this, the data set should look like the following

```bash
---EECS545-Project
 |
 |---data
 |     |
 |     |-tsp15
 |     |    |-instances
 |     |    |   |-test_15n
 |     |    |   |-train_15n
 |     |    |   ...
 |     |    |   
 |     |    |-samples
 |     |        |-test
 |     |        |-train
 |     |        |-valid
 |     |-tsp20
 |     |-tsp25
 |
 |--- ...
 .
 .
 .
```



### `03_train_gnn_imitation.py`

### `04_train_gnn_reinforce.py`