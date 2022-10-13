# ***Travel the Same Path(TSP)***: A Novel TSP Solving Strategy
The paper can be found in [here](https://arxiv.org/abs/2210.05906).

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

```
---Travel-the-Same-Path
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

where `-n`, `-g` are the same flag as we saw, and `-l` is to specify which model we want to evaluate. For example, if we want to evaluate the model which is only trained by imitation learning on `<im_tsp_size>`, then the argument of `-l` should be `<im_tsp_size>` namely the size of tsp instances the model you want to load is trained on. 

## Note

Firstly, the error detection of argument is not implemented, if you pass in things like `-n -100`, some weird things will happen...

