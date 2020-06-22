---
layout:     post
title:      Topological Sort
date:       2020-06-21
tags: [sort, topological]
categories: 
- algorithm
---
A [topological sort](https://bit.ly/2zVkzK4) or topological ordering of a directed graph is a linear ordering of its vertices such that for every directed edge `uv` from vertex `u` to vertex `v`, `u` comes before `v` in the ordering.

The usual algorithms for topological sorting have running time linear in the number of nodes plus the number of edges, asymptotically, `O(|V| + |E|)`

## Kahn's Algorithm
This algorithms works by choosing vertices in the same order as the eventual topological sort. First, find a list of "source nodes" which have no zero indegree (i.e., no incoming edge) and insert them into a stack `stack`; at least one such node must exist in a non-empty acyclic graph.

Then 
```python
def kahn(graph, indegree):
    '''Kahn's topological sorting algorithm, https://bit.ly/2zVkzK4 .
    Parammeters:
    - graph, Adjacency matrix representing a graph
    - indegree, the number of edges directed into a vertex
    '''
    sorted_nodes = []
    stack = [node for node in range(len(graph)) if indegree[node] == 0]
    while stack:
        u = stack.pop()
        sorted_nodes.append(u)
        for v in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                stack.append(v)

    """ If the graph is a DAG, a solution will be contained in the list sorted_nodes.
    Otherwise, the graph must have at least one cycle and therefore a topological sort 
    is impossible."""

    return sorted_nodes if len(sorted_nodes) == len(graph) else []
```