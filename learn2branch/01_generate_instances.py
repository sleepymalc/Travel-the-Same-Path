import os
import argparse
import numpy as np
from docplex.mp.model import Model


def generate_tsp(n, filename):
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
    random = np.random
    random.seed(545)
    coord_x = random.rand(n) * 100
    coord_y = random.rand(n) * 100
    distancia = {(i, j): np.hypot(
        coord_x[i] - coord_x[j], coord_y[i] - coord_y[j]) for i, j in edges}
    mdl = Model('TSP')
    x = mdl.binary_var_dict(edges, name='x')
    d = mdl.continuous_var_dict(cities, name='d')
    mdl.minimize(mdl.sum(distancia[i]*x[i] for i in edges))
    for c in cities:
        mdl.add_constraint(
            mdl.sum(x[(i, j)] for i, j in edges if i == c) == 1, ctname='out_%d' % c)
        mdl.add_constraint(
            mdl.sum(x[(i, j)] for i, j in edges if j == c) == 1, ctname='in_%d' % c)
    for i, j in edges:
        if j != 0:
            mdl.add_indicator(x[(i, j)], d[i]+1 == d[j],
                              name='order_(%d,_%d)' % (i, j))
    print(filename)
    mdl.export_as_lp(filename)

    # with open(filename, 'w') as lp_file:
    #     lp_file.write(
    #         "maximize\nOBJ:" + "".join([f" + 1 x{node+1}" for node in range(len(graph))]) + "\n")
    #     lp_file.write("\nsubject to\n")
    #     for count, group in enumerate(inequalities):
    #         lp_file.write(
    #             f"C{count+1}:" + "".join([f" + x{node+1}" for node in sorted(group)]) + " <= 1\n")
    #     lp_file.write(
    #         "\nbinary\n" + " ".join([f"x{node+1}" for node in range(len(graph))]) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['tsp'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed (default 0).',
        # type=utilities.valid_seed,
        default=0,
    )
    parser.add_argument(
        '-n', '--num',
        help='Number of nodes',
        type=int,
        default=50,
    )
    args = parser.parse_args()

    if args.problem == 'tsp':
        num = int(args.num)

        filenames = []
        nums = []

        n = 1
        # train instances
        n = 10000
        lp_dir = f'data/instances/tsp/train_{num}n'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend(
            [os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nums.extend([num] * n)

        # validation instances
        n = 2000
        lp_dir = f'data/instances/tsp/valid_{num}n'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend(
            [os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nums.extend([num] * n)

        # small transfer instances
        n = 100
        num = int(args.num / 2)
        lp_dir = f'data/instances/tsp/transfer_{num}n'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend(
            [os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nums.extend([num] * n)

        # medium transfer instances
        n = 100
        num = int(args.num / 1.5)
        lp_dir = f'data/instances/tsp/transfer_{num}n'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend(
            [os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nums.extend([num] * n)

        # big transfer instances
        n = 100
        num = int(args.num * 2)
        lp_dir = f'data/instances/tsp/transfer_{num}n'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend(
            [os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nums.extend([num] * n)

        # test instances
        n = 2000
        num = int(args.num)
        lp_dir = f'data/instances/tsp/test_{num}n'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend(
            [os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nums.extend([num] * n)

        # actually generate the instances
        for filename, num in zip(filenames, nums):
            print(f'  generating file {filename} ...')
            generate_tsp(n=num, filename=filename)

        print('done.')
    else:
        raise NotImplementedError
