---
layout: post
title: Minimum Spanning Tree
date: 2020-09-14
tags: prim mst kruskal
categories: algorithm
author: GaoangLau
---
* content
{:toc}


最小生成树问题(Minimum Spanning Tree, MST)：给定一个加权连通图，求出其最小生成树，即权值和最小的连通子图。






Formally, 给定无向连通图 $$ G = (V, E, W) $$，其中 $$ V = \{ v_i \vert 0 \leq i \leq N \} $$，$$ W = \{ w \vert w \geq 0\}$$，$$ E = \{(u, v, w) \vert u, v \in V, w \in W\} $$ 分别表示顶点集合、边的权重及边集合，记 $$ \omega(e) $$ 为边 $$e \in E$$ 的权重。 最小生成树问题即求解 $$ \text{arg min}_{T} \sum_\limits{e \in T} \omega(e) $$，即寻找 $$G$$ 的连通子图 $$T = (V, E' \subset E, W)$$ 使得 $$ \sum_\limits{e \in E'} \omega(e) $$ 最小。

## Prim 算法
Prim 算法过程:
1. 将图的顶点 $$V$$ 分为两类，一类表示在查找过程中已经加入到最小生成树中的顶点集合(记为 $$A$$：初始为任意一个结点 $$ u $$ 构成的单子集)，另一类表示尚未加入到最小生成树中的顶点集合(记为 $$ B $$ ：初始为 $$V \backslash \{u\}$$)； 
2. 从图中寻找一条权重最小的切割边(cut edge)$$^{[1]}$$，将边上位于 $$ B $$ 上的结点加入到 $$ A $$，然后从 $$ B $$ 中删除； 
3. 重复迭代过程 2，直到 $$ B $$ 为空，即所有结点都被加入到最小生成树中。

> [1] 顶点集合 $$ A $$ 与 $$ B $$ 的一条切割边是一条结点分别在 $$ A $$ 与 $$ B $$ 上的边

### 算法正确性
证明: 假设 $$G$$ 是由 Prim 算法生成的一棵树，而 $$H \ne G$$ 为所有最小生成树中与 $$G$$ 有最长公共前缀的一棵树。

记 $$E=(e_1, e_2, ..., e_{n-1})$$ 为算法依次选择的边，$$H_1 = (e_1, ..., e_i)$$ 为 $$G, H$$ 最长公共前缀，则 $$ H_2 = H - H_1$$ 与 $$H_1$$ 分别构成了 $$H$$ 的两个强连通分量。记 $$W$$ 为 $$H_1$$ 中结点构成的集合。

令 $$e_{i+1}=(u, v)$$，记 $$f=(x, y) ^{[1]}$$ 为从 $$H_1$$ 到 $$H_2$$ 上的边且有 $$x \in W$$，则显然有 $$f \ne e_j, 0\leq j \leq i \wedge y \not \in W$$。 考虑:
1. $$\omega(e_{i+1}) > \omega(f)$$，由 Prim 算法的贪心策略可知算法在第 $$i+1$$ 步应该选择 $$f$$ 而不是 $$e_{i+1}$$，与假设 $$G$$ 是由 Prim 算法生成的树矛盾； 
2. $$\omega(e_{i+1}) < \omega(f)$$，则树 $$T=H - \{f\} + \{e_{i+1}\}$$ 也是图的一棵生成树，且权重比 $$H$$ 更小，与 $$H$$ 是 MST 矛盾; 
3. $$\omega(e_{i+1}) = \omega(f)$$，则树 $$T=H - \{f\} + \{e_{i+1}\}$$ 是图的一棵生成树且 $$E' = (e_1, ..., e_{i+1})$$ 是 $$G,T$$ 的公共前缀，与 $$H$$ 是与 $$G$$ 存在最长公共前缀的 MST 矛盾。

> [2] $$x,u$$ 未必是同一个结点 

### 时间复杂度

- 使用邻接矩阵图存储边时，寻找所有最小权边需要 $$ O(\vert V \vert ^2) $$ 运行时间。
- 如果在寻找最小权边过程中引入优先级队列(如下面 `C++` 算法)$$^{[3]}$$，则时间可以优化为 $$O( \vert E \vert \log( \vert V \vert)$$

> [3] 典型的以空间换时间的思路

### C++ 实现
```c++

// This class represents a directed graph using
// adjacency list representation
class Prim {
  std::vector<std::vector<pair<int, int>>> adj;
  // Labels to encode node into 0, 1, 2, ...
  std::unordered_map<int, int> labels;

public:
  int cnt;
  Prim() { cnt = 0; }
  Prim(int N) {cnt=0, adj.reserve(N); }

  int get_label(int u) {
    if (labels.count(u) > 0) return labels[u];
    labels[u] = cnt;
    cnt++;
    return labels[u];
  }

  void add_edge(int u, int v, int w) {
    int lu = get_label(u), lv = get_label(v);
    while (adj.size() <= max(lu, lv) + 1) adj.push_back({});
    adj[lu].push_back(make_pair(lv, w));
    adj[lv].push_back(make_pair(lu, w));
  }

  int cost_of_mst() {
    // Create a priority queue to store std::vector<int>sited vertices.
    priority_queue<std::pair<int, int>, vector<std::pair<int, int>>, greater<std::pair<int, int>>> pq;

    int cost = 0, src = 0; // Taking vertex 0 as source

    // Vector `parent` is used to  print out Tree edges from source 0
    // Vector `key` for storing weights to calculate cost
    vector<int> key(cnt, INT_MAX), parent(cnt, -1);
    vector<bool> visited(cnt, false);

    pq.push(make_pair(0, src));
    key[src] = 0;

    int res = 0;
    /* Looping till priority queue becomes empty */
    while (!pq.empty()) {
      int w = pq.top().first, u = pq.top().second;
      pq.pop();
      if (visited[u]) continue;
      visited[u] = true;
      res += w;

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

    // If key of some node is still int_max, then this is not a SCC.
    for (int i = 0; i < cnt; ++i)
      if (key[i] == INT_MAX) return -1;
    // cost += key[i], printf("%d - %d\n", key[i], i);
    return res;
  }
};
```


## Kruskal 算法
思想 
1. 将图 $$G$$ 中所有的边按权值从小到大排列
2. 新建图 $$G'$$，包含原图 $$G$$ 所有结点，但不含边
3. 从权值最小的边开始，如果这条边连接的两个结点在图 $$G'$$ 中不属于同一个连通分量(可通过并查集来判定)，则添加这条边到图 $$G'$$ 上 
4. 重复过程3，走到图 $$G'$$ 所有结点都在同一个连通分量中

### 时间复杂度
- 边按权值排序储存到优先级队列 $$O(\vert E \vert \log (\vert E \vert)) ^{[3]}$$，算法迭代 $$O(\vert E \vert \log (\vert E \vert))$$

[3] 算法使用 `priority_queue` 以权重为优先级来存储边


### C++ 实现
```c++
#include <deque>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <list>
using namespace std;
#define INF 0x3f3f3f3f

class Kruskal {
  priority_queue<pair<int, pair<int, int>>> pq;
  unordered_map<int, int> labels;

public:
  int cnt=0;;
  Kruskal() { }
  Kruskal(int N) {}

  int get_label(int u) {
    if (labels.count(u) > 0) return labels[u];
    return (labels[u] = cnt++);
  }

  void add_edge(int u, int v, int w) {
    int lu = get_label(u), lv = get_label(v);
    pq.push({-1 * w, make_pair(lu, lv)});
  }

  int find(int u, vector<int> &parents) {
    if (u != parents[u]) parents[u] = find(parents[u], parents);
    return parents[u];
  }

  void merge(int u, int v, vector<int> &parents) {
    int pu = find(u, parents), pv = find(v, parents);
    parents[pu] = parents[pv] = min(pu, pv);
  }

  int cost_of_mst() {
    int cost = 0;
    vector<int> parents(cnt, 0);
    std::iota(parents.begin(), parents.end(), 0);
    vector<pair<int, pair<int, int>>> mst;

    while (!pq.empty()) {
      pair<int, pair<int, int>> top = pq.top();
      pq.pop();
      int w = top.first, u = top.second.first, v = top.second.second;

      int pu = find(u, parents), pv = find(v, parents);
      if (pv != pu) {
        cost -= w;
        merge(u, v, parents);
        mst.push_back(top);
      }
    }
    return cost;
  }
};
```

## References 
- [港大 Minimum Spanning Trees and Prim’s Algorithm](https://home.cse.ust.hk/~dekai/271/notes/L07/L07.pdf)
- [Wikipedia Kruskal Algorithm](https://en.wikipedia.org/wiki/Kruskal%27s_algorithm)
- [Wikipedia Prim Algorithm](https://en.wikipedia.org/wiki/Prim%27s_algorithm)
- [Proof of Prime Algorithm](https://www.uncg.edu/cmp/faculty/srtate/330.f16/primsproof.pdf)