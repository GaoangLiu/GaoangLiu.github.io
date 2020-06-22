import os


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
