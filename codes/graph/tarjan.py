import os
from collections import defaultdict
from typing import List

edges = [[1, 3], [3, 2], [2, 1],                    # SCC 1
         [3, 4],                                    # SCC 2
         [4, 5], [5, 6], [6, 7], [7, 8], [8, 5]]    # SCC 3

graph = defaultdict(list)
for u, v in edges:
    graph[u].append(v)


class Tarjan():
    def __init__(self, graph):
        self.graph = graph
        self.time = 0

    def scc(self)->List[int]:
        '''Finding all strong connected components(SCC), and return each SCC a list of nodes
        '''
        disc = defaultdict(lambda: -1)
        low = defaultdict(lambda: -1)
        visited = defaultdict(lambda: False)
        stack = []
        self.sccs = []  # store all the SCCs
        for n in self.graph.keys():
            if disc[n] == -1:
                self.find_scc(n, disc, low, visited, stack)
        return self.sccs

    def find_scc(self, u, disc, low, visited, stack):
        disc[u] = low[u] = self.time
        self.time += 1
        stack.append(u)
        visited[u] = True
        for v in self.graph.get(u, []):
            if disc[v] == -1:  # when v is not visited
                self.find_scc(v, disc, low, visited, stack)
                low[u] = min(low[u], low[v])
            elif visited[v]:
                ''' In this case, we've found a back edge from u to v, i.e., v is ancestor node of u.
                There may be multiple back edges in subtree taking us to different ancestors,
                then we take the one with minimum `disc` value. E.g., the following node 4 has two
                ancestors, 2 and 3. Node 2 has (possibly) a smaller disc value.
                1 -> 2 -> 3 -> 4
                     ⬆___⬆____|
                '''
                low[u] = min(low[u], disc[v])

        if low[u] == disc[u]:
            _scc = [-1]
            while _scc[-1] != u:
                _scc.append(stack.pop())
                visited[_scc[-1]] = False
            self.sccs.append(_scc[1:])

        return


if __name__ == "__main__":

    t = Tarjan(graph)
    print(t.scc())
