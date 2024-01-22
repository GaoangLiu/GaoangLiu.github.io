---
layout: post
title: Kosaraju's strongly connected components algorithm
date: 2020-06-11
tags: algorithm kosaraju scc
categories: algorithm
author: GaoangLiu
---
* content
{:toc}


[Kosaraju's algorithm](https://en.wikipedia.org/wiki/Kosaraju%27s_algorithm) is a DFS-based **linear time** algorithm to find the strongly connected components (SCC) of a directed graph.



This algorithm makes use of the fact that the **transpose graph** (the same graph with the direction of every edge reversed) has exactly the same strongly connected components as the original graph.

Ideas:
* **First DFS**:  Do a DFS on the original graph, keeping track of the finish times of each node, which can be done with a stack (the source vertex is pushed into a stack only after a DFS finishes, i.e., its children nodes pushed into stack earlier). This way node with highest finishing time will be on top of the stack.

* **Reverse graph**: Reverse the graph using an adjacency list

* **Second DFS**: Do DFS on the reversed graph, with the source vertex as the vertex on top of the stack. When DFS finishes, all nodes visited will form one SCC. If any more nodes remain unvisited, this means there are more SCCs, so pop vertices from top of the stack until a valid unvisited node is found. This will have the highest finishing time of all currently unvisited nodes. This step is repeated until all nodes are visited.

# Examples
There are two SCCs in the following graph: SCC1 [1,2,3], SCC2 [4,5,6]. 

1. Assume we start from node 1, the first DFS results in a stack `st = [6, 5, 4, 3, 2, 1]`. 
<img src="https://i.loli.net/2020/06/11/yhr5H2uW1UsD4it.png" width='500px' alt='Example graph'>    

2. Then we reverse the graph, changing it into:
<img src="https://i.loli.net/2020/06/11/Lun8vSp65A47IBH.png" width='350px' alt='Transpose graph'>    

3. Then we conduct the second DFS from node 1. Node 3, 2 will be visited sequentially before we met node 1 again. Since node 1 was visited already, meaning we have find the first SCC: `[1, 2, 3]`. 
    * The top three nodes `[1, 2, 3]` were popped from stack as there were visited already
    * Algorithm then starts from node 4 and repeats the above procedure until stack is empty.

# Complexity
1. Time complexity, `O(|V| + |E|)`. DFS x 2, each node is visited at most twice.
2. Space complexity, `O(|V|)` since we use a `stack` to store nodes and an array `visited` to keep track of status of nodes

For comparison, [Tarjan's algorithm]({{site.baseurl}}/archives/Tarjan's-strongly-connected-components-algorithm.html) conducts only one DFS and visits each node once, but uses three auxiliary arrays (`low, disc, visited`). More specifically, 

|             | Tarjan        | Kosaraju  |
| ------------|:-------------:|:-----:|
| Time        | `|V| + |E|`   | `2 * (|V| + |E|)` |
| Space       | `3 * |V|`     |   `2 * |V|` |


# Implementation

Python3 
```python
import os
from collections import defaultdict
from typing import List

edges = [[1, 2], [2, 3], [3, 1],            # SCC 1
         [3, 4], [4, 5], [5, 6], [6, 4]]    # SCC 2


class Kosaraju():
    def __init__(self, edges):
        self.graph = defaultdict(list)
        self.transpose = defaultdict(list)
        self.sccs = []
        for u, v in edges:
            self.graph[u].append(v)
            self.transpose[v].append(u)

    def dfs_1(self, node, visited, stack):
        '''First DFS to push nodes into stack
        '''
        if not visited[node]:
            visited[node] = True
            f = lambda v: not visited[v] and self.dfs_1(v, visited, stack)
            for v in self.graph[node]:
                f(v)
            stack.append(node)

    def dfs_2(self, node, visited):
        '''Second DFS to search SCC and mark nodes as visited
        '''
        self.sccs[-1].append(node)
        visited[node] = True
        for v in self.transpose[node]:
            if not visited[v]:
                self.dfs_2(v, visited)

    def search_all_sccs(self)->List[List[int]]:
        '''Find all SCCs and return them as a list.
        '''
        cnt = max(self.graph.keys()) + 1
        visited = [False] * cnt
        stack = []
        for k in self.graph:
            self.dfs_1(k, visited, stack)
        print(stack)

        visited = [False] * cnt
        while stack:
            node = stack.pop()
            if not visited[node]:
                self.sccs.append([])
                self.dfs_2(node, visited)
        return

if __name__ == "__main__":
    k = Kosaraju(edges)
    k.search_all_sccs()
    print(k.sccs)
```