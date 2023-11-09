from graphviz import Digraph
'''More examples on how to draw graph can be found here:
https://graphviz.readthedocs.io/en/stable/examples.html#process-py
'''

f = Digraph(node_attr={'color': 'lightblue2', 'style': 'filled'}, filename='dag.gv')

# use rankdir = 'LR' to show nodes from left to right (default is TB: top-bottom)
f.attr(size='8,5', rankdir='LR')
edges = [(1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 6), (6, 4)]

for n in range(1, 7):
    f.node(str(n), str(n))

for e in edges:
    u, v = str(e[1]), str(e[0])
    f.edge(u, v)

f.view()
f.render('dag.png')