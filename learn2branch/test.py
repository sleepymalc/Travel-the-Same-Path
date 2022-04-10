from fileinput import filename
import numpy as np
from docplex.mp.model import Model

n = 10
filename = 'instance_1.lp'
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

mdl.export_as_lp(filename)
