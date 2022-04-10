import os
import argparse
import numpy as np
import scipy.sparse
import utilities
from itertools import combinations


class Graph:
    """
    Container for a graph.

    Parameters
    ----------
    number_of_nodes : int
        The number of nodes in the graph.
    edges : set of tuples (int, int)
        The edges of the graph, where the integers refer to the nodes.
    degrees : numpy array of integers
        The degrees of the nodes in the graph.
    neighbors : dictionary of type {int: set of ints}
        The neighbors of each node in the graph.
    """

    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    def __len__(self):
        """
        The number of nodes in the graph.
        """
        return self.number_of_nodes

    def greedy_clique_partition(self):
        """
        Partition the graph into cliques using a greedy algorithm.

        Returns
        -------
        list of sets
            The resulting clique partition.
        """
        cliques = []
        leftover_nodes = (-self.degrees).argsort().tolist()

        while leftover_nodes:
            clique_center, leftover_nodes = leftover_nodes[0], leftover_nodes[1:]
            clique = {clique_center}
            neighbors = self.neighbors[clique_center].intersection(
                leftover_nodes)
            densest_neighbors = sorted(
                neighbors, key=lambda x: -self.degrees[x])
            for neighbor in densest_neighbors:
                # Can you add it to the clique, and maintain cliqueness?
                if all([neighbor in self.neighbors[clique_node] for clique_node in clique]):
                    clique.add(neighbor)
            cliques.append(clique)
            leftover_nodes = [
                node for node in leftover_nodes if node not in clique]

        return cliques

    @staticmethod
    def erdos_renyi(number_of_nodes, edge_probability, random):
        """
        Generate an Erdös-Rényi random graph with a given edge probability.

        Parameters
        ----------
        number_of_nodes : int
            The number of nodes in the graph.
        edge_probability : float in [0,1]
            The probability of generating each edge.
        random : numpy.random.RandomState
            A random number generator.

        Returns
        -------
        Graph
            The generated graph.
        """
        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for edge in combinations(np.arange(number_of_nodes), 2):
            if random.uniform() < edge_probability:
                edges.add(edge)
                degrees[edge[0]] += 1
                degrees[edge[1]] += 1
                neighbors[edge[0]].add(edge[1])
                neighbors[edge[1]].add(edge[0])
        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

    @staticmethod
    def barabasi_albert(number_of_nodes, affinity, random):
        """
        Generate a Barabási-Albert random graph with a given edge probability.

        Parameters
        ----------
        number_of_nodes : int
            The number of nodes in the graph.
        affinity : integer >= 1
            The number of nodes each new node will be attached to, in the sampling scheme.
        random : numpy.random.RandomState
            A random number generator.

        Returns
        -------
        Graph
            The generated graph.
        """
        assert affinity >= 1 and affinity < number_of_nodes

        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for new_node in range(affinity, number_of_nodes):
            # first node is connected to all previous ones (star-shape)
            if new_node == affinity:
                neighborhood = np.arange(new_node)
            # remaining nodes are picked stochastically
            else:
                neighbor_prob = degrees[:new_node] / (2*len(edges))
                neighborhood = random.choice(
                    new_node, affinity, replace=False, p=neighbor_prob)
            for node in neighborhood:
                edges.add((node, new_node))
                degrees[node] += 1
                degrees[new_node] += 1
                neighbors[node].add(new_node)
                neighbors[new_node].add(node)

        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph


def generate_setcover(nrows, ncols, density, filename, rng, max_coef=100):
    """
    Generates a setcover instance with specified characteristics, and writes
    it to a file in the LP format.

    Approach described in:
    E.Balas and A.Ho, Set covering algorithms using cutting planes, heuristics,
    and subgradient optimization: A computational study, Mathematical
    Programming, 12 (1980), 37-60.

    Parameters
    ----------
    nrows : int
        Desired number of rows
    ncols : int
        Desired number of columns
    density: float between 0 (excluded) and 1 (included)
        Desired density of the constraint matrix
    filename: str
        File to which the LP will be written
    rng: numpy.random.RandomState
        Random number generator
    max_coef: int
        Maximum objective coefficient (>=1)
    """
    nnzrs = int(nrows * ncols * density)

    assert nnzrs >= nrows  # at least 1 col per row
    assert nnzrs >= 2 * ncols  # at leats 2 rows per col

    # compute number of rows per column
    indices = rng.choice(ncols, size=nnzrs)  # random column indexes
    # force at leats 2 rows per col
    indices[:2 * ncols] = np.repeat(np.arange(ncols), 2)
    _, col_nrows = np.unique(indices, return_counts=True)

    # for each column, sample random rows
    indices[:nrows] = rng.permutation(nrows)  # force at least 1 column per row
    i = 0
    indptr = [0]
    for n in col_nrows:

        # empty column, fill with random rows
        if i >= nrows:
            indices[i:i+n] = rng.choice(nrows, size=n, replace=False)

        # partially filled column, complete with random rows among remaining ones
        elif i + n > nrows:
            remaining_rows = np.setdiff1d(
                np.arange(nrows), indices[i:nrows], assume_unique=True)
            indices[nrows:i +
                    n] = rng.choice(remaining_rows, size=i+n-nrows, replace=False)

        i += n
        indptr.append(i)

    # objective coefficients
    c = rng.randint(max_coef, size=ncols) + 1

    # sparce CSC to sparse CSR matrix
    A = scipy.sparse.csc_matrix(
        (np.ones(len(indices), dtype=int), indices, indptr),
        shape=(nrows, ncols)).tocsr()
    indices = A.indices
    indptr = A.indptr

    # write problem
    with open(filename, 'w') as file:
        file.write("minimize\nOBJ:")
        file.write("".join([f" +{c[j]} x{j+1}" for j in range(ncols)]))

        file.write("\n\nsubject to\n")
        for i in range(nrows):
            row_cols_str = "".join(
                [f" +1 x{j+1}" for j in indices[indptr[i]:indptr[i+1]]])
            file.write(f"C{i}:" + row_cols_str + f" >= 1\n")

        file.write("\nbinary\n")
        file.write("".join([f" x{j+1}" for j in range(ncols)]))


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
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    if args.problem == 'tsp':
        nrows = 500
        ncols = 1000
        dens = 0.05
        max_coef = 100

        filenames = []
        nrowss = []
        ncolss = []
        denss = []

        # train instances
        n = 1000
        lp_dir = f'data/instances/setcover/train_{nrows}r_{ncols}c_{dens}d'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend(
            [os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nrowss.extend([nrows] * n)
        ncolss.extend([ncols] * n)
        denss.extend([dens] * n)

        # validation instances
        n = 200
        lp_dir = f'data/instances/setcover/valid_{nrows}r_{ncols}c_{dens}d'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend(
            [os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nrowss.extend([nrows] * n)
        ncolss.extend([ncols] * n)
        denss.extend([dens] * n)

        # small transfer instances
        n = 10
        nrows = 50
        lp_dir = f'data/instances/setcover/transfer_{nrows}r_{ncols}c_{dens}d'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend(
            [os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nrowss.extend([nrows] * n)
        ncolss.extend([ncols] * n)
        denss.extend([dens] * n)

        # medium transfer instances
        n = 10
        nrows = 100
        lp_dir = f'data/instances/setcover/transfer_{nrows}r_{ncols}c_{dens}d'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend(
            [os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nrowss.extend([nrows] * n)
        ncolss.extend([ncols] * n)
        denss.extend([dens] * n)

        # big transfer instances
        n = 10
        nrows = 200
        lp_dir = f'data/instances/setcover/transfer_{nrows}r_{ncols}c_{dens}d'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend(
            [os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nrowss.extend([nrows] * n)
        ncolss.extend([ncols] * n)
        denss.extend([dens] * n)

        # test instances
        n = 200
        nrows = 50
        ncols = 100
        lp_dir = f'data/instances/setcover/test_{nrows}r_{ncols}c_{dens}d'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend(
            [os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nrowss.extend([nrows] * n)
        ncolss.extend([ncols] * n)
        denss.extend([dens] * n)

        # actually generate the instances
        for filename, nrows, ncols, dens in zip(filenames, nrowss, ncolss, denss):
            print(f'  generating file {filename} ...')
            generate_setcover(nrows=nrows, ncols=ncols, density=dens,
                              filename=filename, rng=rng, max_coef=max_coef)

        print('done.')
