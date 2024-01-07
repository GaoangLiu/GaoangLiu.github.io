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
This algorithm works by choosing vertices in the same order as the eventual topological sort. First, find a list of "source nodes" which have zero indegree (i.e., no incoming edge) and insert them into a stack `stack`; at least one such node must exist in a non-empty acyclic graph.

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


## Topological sort with DFS 
Another way to conduct Topological sort is [DFS](https://raw.githubusercontent.com/gaonagliu/figures/master/codes/dfs_topological_sort.py) (or BFS). 

```python
def dfs_topological_sort(arr, n):
    """ Topological sort with DFS. Return an empty list if 
    there is a cycle. 
    """
    graph = [[] for _ in range(n)]
    for u, v in arr:
        graph[u].append(v)

    visited, stack = [0] * n, []

    def dfs(u):
        if visited[u] == -1:
            return False
        if visited[u] == 1:
            return True
            
        visited[u] = -1
        for v in graph[u]:
            if not dfs(v):
                return -1
        stack.append(u)
        visited[u] = 1
        return True

    for u in range(n):
        if not dfs(u):
            return []
    return stack[::-1]
```
We use `visited` keep in track the status of a node:
1. if node `u` has not been visited, then mark it as 0.
2. if node `u` is being visited, then mark it as -1. If we find a vertex marked as -1 in DFS, then their is a ring.
3. if node `u` has been visited, then mark it as 1. If a vertex was marked as 1, then no ring contains v or its successors.