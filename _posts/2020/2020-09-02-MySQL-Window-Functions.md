---
layout: post
title: MySQL Window Functions
date: 2020-09-02
tags: mysql window_function
categories: sql
author: GaoangLiu
---
* content
{:toc}



SQL 处理的基本单位不是记录，而是集合。 —— Joe Celko 







窗口函数与聚合函数的区别联系：
1. 窗口函数并没有将记录聚合在一起，执行完之后记录个数不变；
2. 聚合函数也可以用做窗口函数

窗口函数的语法: 

```bash 
WIN_FUNC OVER (
    PARTITION BY <expression>[{,<expression>...}]
    ORDER BY <expression> [ASC|DESC], [{,<expression>...}]
)
```
其中 `parition by` 按指定字段进行分组(即记录的一个子集合)，`order by` 按指定字段进行排序。 

使用场景：业务需求分组并在组内进行排行比较。

## `ranking` 序号函数 

主要有 `row_number()`, `rank()`, `dense_rank()`, `ntile()` 几个函数 

### `ntile(n)` 函数 
对组内记录记录按给定字段排序后进行分组，并赋以组号。示例

```sql
ntile(3) over (partition by id order by date) as t_ntile
```

* 将已经按 `id` 分割并按 `date` 排序的每组数据分成 3 组。
* 如果 `partition` 没有指定，则将所有记录分为 3 组。
* 如果 `order` 没有指定，则每组数据随机分成 3 组 (SQL 中记录本质上是集合，记录之间没有先后顺序而言)

### `row_number()`
对组内记录按指定字段排序，**严格递增地**赋以行数。示例

```sql
row_number() over (partition by id order by date) as t_row
```

* 将已经按 `id` 分割并按 `date` 排序的记录按组从 1 开始编号
    * 无论按什么字段排序或者不排序，组内每条记录的行号都不会相同 
* 如果 `partition` 没有指定，则对所有记录按 `date` 顺序进行编号
    

### `rank()`
对组内记录按指定字段排序，**非递减**赋以序号。示例

```sql
rank() over (partition by id order by date) as t_rank
```

* 将已经按 `id` 分割并按 `date` 排序的记录按组从 1 开始编号
    * 特别的，如果 `partition` 与 `order` 指定的为同一字段，那么所有序号都为 1
* 如果 `partition` 没有指定，则对所有记录按 `date` 顺序进行编号
    * 序号之间值可能不连续，当 K 条记录存在相同的行号时，下一个 `date` 不同的记录的行号与当前行号相差为 K
* 如果 `order` 没有指定字段，则组内所有序号都为 1
* 如果 `partition`, `order` 都没有指定，则所有序号都为 1

### `dense_rank()`
与 `rank()` 类似，唯一区别在于组内编号在数值上是连续的，相差至多为 1. 

### `percent_rank()`
percent_rank = $$(rank - 1) / (rows - 1)$$，基中 rank 为 RANK() 函数产生的序号，rows 为当前窗口的记录总行数。

### `cume_dist()`
分组内小于等于当前 rank 值的行数比上分组内总行数。


## 滑动窗口
计算当前记录之前所有 salary 之和

```sql 
SUM(salary) OVER (ORDER BY employee_id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as `running total` 
-- or simply 
-- SUM(salary) OVER (ORDER BY employee_id ROWS UNBOUNDED PRECEDING) as `running total` or 
```

计算窗口长度为 3 的 salary 之和
```sql 
SUM(salary) OVER (ORDER BY employee_id ROWS 2 PRECEDING) as `running total 3`
```

### `lag` and `lead`
语法 
```sql 
LAG/LEAD (<expression>[,offset[, default_value]]) OVER (
    PARTITION BY expr,...
    ORDER BY expr [ASC|DESC],...
)
```

`lag` 返回之前记录的值，而  `lead` 返回之后记录的值。示例，返回下一条记录 `salary` 的值，最后一条 lead 返回值设置为 0

```sql 
LEAD(salary, 1, 0) OVER (ORDER BY employee_id) AS `lead salary`.
-- LAG(salary) OVER (ORDER BY employee_id) AS `lag salary`.
```

## 头尾函数 
* `first_value()` 按指定字段返回组内第一条记录的值 
* `last_value()` 按指定字段返回组内最后一条记录的值 


--- 
To be improved...

