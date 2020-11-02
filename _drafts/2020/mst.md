---
layout:     post
title:      Minimum Spanning Tree
date:       2020-09-14
tags: [prim, mst, kruskal]
categories: 
- algorithm
---

最小生成树问题(Minimum Spanning Tree, MST)：给定一个加权连通图，求出其最小生成树，即权值和最小的连通子图。



Formally, 给定无向连通图 $$ G = (V, E, W) $$，其中 $$ V = \{ v_i \vert 0 \leq i \leq N \} $$，$$ W = \{ w \vert w \geq 0\}$$，$$ E = \{(u, v, w) \vert u, v \in V, w \in W\} $$ 分别表示顶点集合、边的权重及边集合，记 $$ \omega(e) $$ 为边 $$e \in E$$ 的权重。 求 $$G$$ 的连通子图 $$T = (V, E' \subset E, W)$$ 使得 $$ \sum_\limits{e \in E'} \omega(e) $$ 最小。

## Prim 算法
Prim 算法过程:
1. 将图的顶点 $$V$$ 分为两类，一类表示在查找过程中已经加入到最小生成树中的顶点集合(记为 $$A$$：初始为任意一个节点 $$ u $$ 构成的单子集)，另一类表示尚未加入到最小生成树中的顶点集合(记为 $$ B $$ ：初始为 $$V \backslash \{u\}$$)； 
2. 从图中寻找一条权重最小的切割边(cut edge)$$^{[1]}$$，将边上位于 $$ B $$ 上的节点加入到 $$ A $$，然后从 $$ B $$ 中删除； 
3. 重复迭代过程 2，直到 $$ B $$ 为空，即所有节点都被加入到最小生成树中。

> [1] 顶点集合 $$ A $$ 与 $$ B $$ 的一条切割边是一条节点分别在 $$ A $$ 与 $$ B $$ 上的边

### 算法正确性
证明: 假设 $$G$$ 是由 Prim 算法生成的一棵树，而 $$H \ne G$$ 为所有最小生成树中与 $$G$$ 有最长公共前缀的一棵树。

记 $$E=(e_1, e_2, ..., e_{n-1})$$ 为算法依次选择的边，$$H_1 = (e_1, ..., e_i)$$ 为 $$G, H$$ 最长公共前缀，则 $$ H_2 = H \ H_1$$ 与 $$H_1$$ 分别构成了 $$H$$ 的两个强连通分量。记 $$W$$ 为 $$H_1$$ 中节点构成的集合。

令 $$e_{i+1}=(u, v)$$，记 $$f=(x, y) ^{[1]}$$ 为从 $$H_1$$ 到 $$H_2$$ 上的边且有 $$x \in W$$，则显然有 $$f \ne e_j, 0\leq j \leq i \wedge y \not \in W$$。 考虑:
1. $$\omega(e_{i+1}) > \omega(f)$$，由 Prim 算法的贪心策略可知算法在第 $$i+1$$ 步应该选择 $$f$$ 而不是 $$e_{i+1}$$，与假设 $$G$$ 是由 Prim 算法生成的树矛盾； 
2. $$\omega(e_{i+1}) < \omega(f)$$，则树 $$T=H - \{f\} + \{e_{i+1}\}$$ 也是图的一棵生成树，且权重比 $$H$$ 更小，与 $$H$$ 是 MST 矛盾; 
3. $$\omega(e_{i+1}) = \omega(f)$$，则树 $$T=H - \{f\} + \{e_{i+1}\}$$ 是图的一棵生成树且 $$E' = (e_1, ..., e_{i+1})$$ 是 $$G,T$$ 的公共前缀，与 $$H$$ 是与 $$G$$ 存在最长公共前缀的 MST 矛盾。

> [2] $$x,u$$ 未必是同一个节点 

### 时间复杂度

- 使用邻接矩阵图存储边时，寻找所有最小权边需要 $$ O(\vert V \vert ^2) $$ 运行时间。
- 如果在寻找最小权边过程中引入优先级队列(如下面 `C++` 算法)$$^{[3]}$$，则时间可以优化为 $$O( \vert E \vert \log( \vert V \vert)$$

> [3] 典型的以空间换时间的思路

### C++ 实现
```c++
#include <iostream>
#include <list>
#include <queue>
#include <vector>
using namespace std;
#define INF 0x3f3f3f3f

typedef pair<int, int> pii;

// This class represents a directed graph using
// adjacency list representation
class Graph {
  int V; // No. of vertices
  list<pair<int, int>> *adj;

public:
  Graph(int V); // Constructor
  void add_edge(int u, int v, int w);
  int cost_of_mst();
};

// Allocates memory for adjacency list
Graph::Graph(int V) {
  this->V = V;
  adj = new list<pii>[V];
}

void Graph::add_edge(int u, int v, int w) {
  adj[u].push_back(make_pair(v, w));
  adj[v].push_back(make_pair(u, w));
}

// Prints shortest paths from src to all other vertices
int Graph::cost_of_mst() {
  // Create a priority queue to store visited vertices.
  // Item is a pair <weight, node id>
  priority_queue<pii, vector<pii>, greater<pii>> pq;

  int cost = 0, src = 0; // Taking vertex 0 as source

  // Vector `parent` is used to  print out Tree edges from source 0
  // Vector `key` for storing weights to calculate cost
  vector<int> key(V, INF), parent(V, -1);
  vector<bool> visited(V, false);

  pq.push(make_pair(0, src));
  key[src] = 0;

  /* Looping till priority queue becomes empty */
  while (!pq.empty()) {
    int w = pq.top().first, u = pq.top().second;
    pq.pop();

    visited[u] = true; // Include vertex in MST

    // 'i' is used to get all adjacent vertices of a vertex
    for (auto &[v, weight] : adj[u]) {
      // Get vertex label and weight of current adjacent of u.
      // If v is not in MST and weight of (u, v) is smaller
      // than current key of v
      if (visited[v] == false && key[v] > weight) {
        // Updating key of v
        key[v] = weight;
        pq.push(make_pair(key[v], v));
        parent[v] = u;
      }
    }
  }
  
  // Print edges of MST using parent array
  for (int i = 1; i < V; ++i)
    cost += key[i], printf("%d - %d\n", parent[i], i);
  return cost;
}
```


## Kruskal 算法
思想 
1. 将图 $$G$$ 中所有的边按权值从小到大排列
2. 新建图 $$G'$$，包含原图 $$G$$ 所有节点，但不含边
3. 从权值最小的边开始，如果这条边连接的两个节点于图 $$G'$$ 中不在同一个连通分量中，则添加这条边到图 $$G'$$ 上
4. 重复过程3，走到图 $$G'$$ 所有节点都在同一个连通分量中

### 时间复杂度
- 边按权值排序储存到优先级队列 $$O(\vert E \vert \log (\vert E \vert)) ^{[3]}$$，算法迭代 $$O(\vert E \vert \log (\vert E \vert))$$

[3] 算法使用 `priority_queue` 以权重为优先级来存储边

### C++ 实现
```c++
#include <iostream>
#include <list>
#include <numeric>
#include <queue>
#include <vector>

using namespace std;
#define INF 0x3f3f3f3f

typedef pair<int, int> pii;

class Graph {
  int V; // No. of vertices
  std::priority_queue<pair<int, pii>> pq;

public:
  Graph(int V);
  void add_edge(int u, int v, int w);
  int kruskal_mst();
};

Graph::Graph(int V) { this->V = V; }

void Graph::add_edge(int u, int v, int w) { pq.push({-1 * w, {u, v}}); }

int find(int u, vector<int> &parents) {
  if (u != parents[u])
    parents[u] = find(parents[u], parents);
  return parents[u];
}

void merge(int u, int v, vector<int> &parents) {
  int pu = find(u, parents), pv = find(v, parents);
  parents[pu] = parents[pv] = min(pu, pv);
}

int Graph::kruskal_mst() {
  int cost = 0;
  std::vector<int> parents(V);
  std::iota(parents.begin(), parents.end(), 0);
  std::vector<pair<int, pii>> mst; // To store edges of minimum spanning tree

  while (!pq.empty()) {
    pair<int, pii> top = pq.top();
    pq.pop();
    int weight = top.first, u = top.second.first, v = top.second.second;

    int pu = find(u, parents), pv = find(v, parents);
    // Decide whether two nodes u, v belong to the same connected component 
    if (pu != pv) {
      cost -= weight; // Note that weights are stored as negative numbers to
                      // leverage min heap
      merge(u, v, parents);
      mst.push_back(top);
    }
  }

  return cost;
}
```

## References 
- [港大 Minimum Spanning Trees and Prim’s Algorithm](https://home.cse.ust.hk/~dekai/271/notes/L07/L07.pdf)
- [Wikipedia Kruskal Algorithm](https://en.wikipedia.org/wiki/Kruskal%27s_algorithm)
- [Wikipedia Prim Algorithm](https://en.wikipedia.org/wiki/Prim%27s_algorithm)
- [Proof of Prime Algorithm](https://www.uncg.edu/cmp/faculty/srtate/330.f16/primsproof.pdf)