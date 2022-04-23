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

1. Create a environment:

   ```bash
   conda create -n ecole python=3.8
   ```

2. Activate the newly created environment:

   ```bash
   conda activate ecole
   ```

   where we use `ecole` as the name of this virtual environment. You can change it to whatever you want. 

3. Install `ecole`, `Pyscipopt`(specifically for `05_evaluate.py`):

   ```bash
   conda conda install -c conda-forge ecole
   conda install -c conda-forge ecole pyscipopt
   ```

4. Install `docplex` 

   ```bash
   pip install docplex
   ```

   **Warning**: You'll need to check your `pip`'s location. Namely, `whcih pip` are you using. This can be checked by using `which pip`. Make sure you're using the `pip` in your virtual environment. If now, specify the prefix of the `pip` as `/home/<uniqname>/anaconda3/envs/ecole/bin/pip`. Same for `pip3` in the next step.

5. Install `pytorch` with propriate CUDA version.

   ```bash
   pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
   ```

6. Install `pytorch_geometric` **without** `torch-spline-conv`. This is a buggy dependency and pyg(pytorch geometric) group decide to give up eventually as well. We simply ignore it since we don't need it anyway. 

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

### Data Generation

Both `01_generate_instances.py` and `02_generate_dataset.py` is for generating data. Please first set the parameters in `parameters.py`, where we specify all the data set parameter there.

#### `01_generate_instances.py`

This is to generate TSP instances. We formulate a random TSP problem by mixed linear programming and record it into `instances_*.lp`, where `*` refer to the number of the instance. To run this, use 

```bash
python 01_generate_instances.py -n <tsp_size>
```

where `<tsp_size>` is `10`, `15`, `20`, `25` in our experiments. 

**Note**: In `01_generate_instances.py`, we also generate some other instances with various `<tsp_size>` for transfer learning. Specifically, for an input `<tsp_size>`, we not only generate testing, training and validation data with `<tsp_size>`, we also generate small, medium, and large transfer data with size `<tsp_size>/2`, `<tsp_size>/1.5`, and `<tsp_size>*2` respectively.

#### `02_generate_dataset.py`

After generating all the instances, we can create the dataset for *imitation learning*. This will record all the branching information when `SCIPY` solving an instance, and we'll use this information to learn the expert choice. To generate the dataset, we run 

```bash
python 02_generate_dataset.py -p <probaility> -n <tsp_size> -j <workers>
```

These flags mean: 

1. `-p`: The probability for `SCIPY` asking for a strong branch. (Since this is un-common to ask for strong branch in every branching decision, and the original paper use `0.05`. Here, we use `0.5` instead.)
2. `-n`: TSP size. In our experiment, we train on `TSP10` and `TSP15` by using imitation learning. 
3. `-j`: Number of parallel jobs when creating data sample for one instance. This depends on your cpu power.

After running this, the data set should look like the following

```json
---EECS545-Project
 |---data
 |     |-tsp15
 |     |    |-instances
 |     |    |   |-test
 |     |    |   |-train
 |     |    |   |-valid
 |     |    |-samples
 |     |        |-train
 |     |        |-valid
 |     |-tsp20
 |     |-tsp25
 |--- ...
 .
 .
 .
```

### Training

#### `03_train_gnn_imitation.py`

We are now ready to train. We first use imitation learning. To run the script, use 

```bash
python 03_train_gnn_imitation.py -n <tsp_size> -g <GPU id>
```

where `-n` is the size you want to train on, and `-g` is to specify the `GPU id` used by `pytorch` with `CUDA`. 

#### `04_evaluate.py`

To evaluate our model performance, we use the following command. 

```bash
python 05_evaluate.py -n <tsp_size> -g <GPU id> -l <imitation size>
```

where `-n`, `-g` are the same flag as we saw, and `-l` is to specify which model we want to evaluate. For example, if one wants to evaluate the model which is only trained by imitation learning on `<im_tsp_size>`, then the argument of `-l` should be `<im_tsp_size>` namely the size of tsp instances the model you want to load is trained on. 

## Note

Firstly, the error detection of argument is not implemented, if you pass in things like `-n -100`, some weird things will happen...

