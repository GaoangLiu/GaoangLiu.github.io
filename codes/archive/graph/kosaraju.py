import os
from collections import defaultdict
from typing import List

edges = [[1, 3], [3, 2], [2, 1],                    # SCC 1
         [3, 4],                                    # SCC 2
         [4, 5], [5, 6], [6, 7], [7, 8], [8, 5]]    # SCC 3

edges = [[1, 2], [2, 3], [3, 1],
         [3, 4], [4, 5], [5, 6], [6, 4]]


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

            def f(v): return not visited[v] and self.dfs_1(v, visited, stack)
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
