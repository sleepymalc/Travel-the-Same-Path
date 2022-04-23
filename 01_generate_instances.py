import os
import numpy as np
import argparse
import parameters
import csv
from docplex.mp.model import Model
#from concorde.tsp import TSPSolver
import time


def generate_tsp(n, filename, random):
    """
    Generate a TSP instance in CPLEX LP format.
    Parameters
    ----------
    graph : number of nodes
        just number of nodes of the TSP instance
    filename : str
        Path to the file to save.
    """
    cities = [i for i in range(n)]
    edges = [(i, j) for i in cities for j in cities if i != j]
    coord_x = random.rand(n) * 100
    coord_y = random.rand(n) * 100
    distances = {(i, j): np.hypot(coord_x[i] - coord_x[j], coord_y[i] - coord_y[j]) for i, j in edges}

    mdl = Model('TSP')
    x = mdl.binary_var_dict(edges, name='x')
    d = mdl.continuous_var_dict(cities, name='d')
    mdl.minimize(mdl.sum(distances[i] * x[i] for i in edges))
    for c in cities:
        mdl.add_constraint(mdl.sum(x[(i, j)] for i, j in edges if i == c) == 1, ctname='out_%d' % c)
        mdl.add_constraint(mdl.sum(x[(i, j)] for i, j in edges if j == c) == 1, ctname='in_%d' % c)
    for i, j in edges:
        if j != 0:
            mdl.add_indicator(x[(i, j)], d[i] + 1 == d[j], name='order_(%d,_%d)' % (i, j))
    print(filename)
    mdl.export_as_lp(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n',
        '--num',
        help='tsp size test on.',
        type=int,
        default=15,
    )

    parser.add_argument(
        '-t',
        '--train_num',
        help='training set number',
        type=int,
        default=10000,
    )

    parser.add_argument(
        '-t',
        '--valid_num',
        help='validation set number',
        type=int,
        default=2000,
    )

    parser.add_argument(
        '-t',
        '--test_num',
        help='testing set number',
        type=int,
        default=100,
    )

    args = parser.parse_args()

    tsp_size = int(args.num)
    num = tsp_size
    seed = parameters.seed
    random = np.random
    random.seed(seed)

    filenames = []
    nums = []

    # test instances
    n = parameters.test_instance
    num = int(args.test_num)
    lp_dir = f'data/tsp{tsp_size}/instances/test_{num}n'
    print(f"{n} instances in {lp_dir}")
    os.makedirs(lp_dir)
    filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
    nums.extend([num] * n)

    # train instances
    n = int(args.train_num)
    lp_dir = f'data/tsp{tsp_size}/instances/train_{num}n'
    print(f"{n} instances in {lp_dir}")
    os.makedirs(lp_dir)
    filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
    nums.extend([num] * n)

    # validation instances
    n = int(args.valid_num)
    lp_dir = f'data/tsp{tsp_size}/instances/valid_{num}n'
    print(f"{n} instances in {lp_dir}")
    os.makedirs(lp_dir)
    filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
    nums.extend([num] * n)

    # actually generate the instances
    for filename, num in zip(filenames, nums):
        print(f'  generating file {filename} ...')
        generate_tsp(n=num, filename=filename, random=random)

    # os.makedirs('results', exist_ok=True)
    # concorde_result = f"concorde_{tsp_size}n_{time.strftime('%m%d-%H%M')}.csv"
    # with open(f"results/{concorde_result}", 'w', newline='') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=['instance', ['walltime']])
    #     # actually generate the instances
    #     for filename, num in zip(filenames, nums):
    #         print(f'  generating file {filename} ...')
    #         cities = [i for i in range(num)]
    #         edges = [(i, j) for i in cities for j in cities if i != j]
    #         coord_x = random.rand(num) * 100
    #         coord_y = random.rand(num) * 100

    #         start_time = time.time()
    #         solver = TSPSolver.from_data(coord_x, coord_y, norm="GEO")
    #         solution = solver.solve()
    #         end_time = time.time() - start_time
    #         writer.writerow({
    #             'instance': filename,
    #             'walltime': end_time,
    #         })
    #         csvfile.flush()

    #         distances = {(i, j): np.hypot(coord_x[i] - coord_x[j], coord_y[i] - coord_y[j]) for i, j in edges}

    #         mdl = Model('TSP')
    #         x = mdl.binary_var_dict(edges, name='x')
    #         d = mdl.continuous_var_dict(cities, name='d')
    #         mdl.minimize(mdl.sum(distances[i] * x[i] for i in edges))
    #         for c in cities:
    #             mdl.add_constraint(mdl.sum(x[(i, j)] for i, j in edges if i == c) == 1, ctname='out_%d' % c)
    #             mdl.add_constraint(mdl.sum(x[(i, j)] for i, j in edges if j == c) == 1, ctname='in_%d' % c)
    #         for i, j in edges:
    #             if j != 0:
    #                 mdl.add_indicator(x[(i, j)], d[i] + 1 == d[j], name='order_(%d,_%d)' % (i, j))
    #         print(filename)
    #         mdl.export_as_lp(filename)

    print('done.')