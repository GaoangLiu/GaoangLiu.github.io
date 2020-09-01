---
layout: post
title: MySQL Cheat-Sheet
date: 2020-08-22
tags: mysql
categories: sql
author: GaoangLau
---
* content
{:toc}




SQL 处理的基本单位不是记录，而是集合。 —— Joe Celko 






A simple [cheatsheet](http://git.io/JUvIJ) on MySQL. 



## 基本语法
* `delete from table [where clause]`, 按条件删除数据
* `left(str, len)`, 截取左边 `len` 长度字符。E.g., `left("abcd", 2)` = `ab`
* `substring(str, pos)`, 从 `pos` 处截取字符, e.g., `substring("abcde", 2)` = `bcde`。注意 `pos` 下标从 1 开始。如果 `pos` 为负值，则从右边开始截取
* `substring(str, pos, len)`, 截取长度为 `len` 的字符串
* `substring_index(str, delim, pos)`, 以 `delim` 为分割符，取第 `pos` 个分割符前面的内容(包含分割符)；如果 `pos` 为负值，则取倒数 `abs(pos)` 个分割符之后的内容。 E.g., `substring_index(123.344.222.222, '.', -3)` = "344.222.222".
* `mid`，是 `substring` 的别名
* `cast(value as dtype)`, 数据类型转换, e.g., `cast('2020-02-02' as date)； cast(150 as char)；`
* `extract(day|month|minute from date)`，从日期/时间中提取分项 
* `length(), char_length()`，分别按字节、字符形式返回字符串长度。 E.g, `select length(_utf8 '和'), char_length(_utf8 '和')` 返回 `3, 1`。对于仅由 「英文字母+数字+标点」 组成的字符串二者没有区别。
  * 注: 当编码格式为 `utf8` （`SET character_set_client=utf8`） 时，中文及特殊符号字节及字符长度为3, 1。对于 `gbk `格式，尽管很多网络资料声明其 `length` 为2，但在实际测试(MacOS/MySQL 8.0.12)中通常不是2，而且不固定。比如下面两个长度均为3的字符串，返回的 length 并不一致:
  *  <img src="http://git.io/JUUj4" width="250px"> <img src="http://git.io/JUUjV" width="250px"> 
* `ucase(), lcase()` 大小写转换。E.g., 首字母大写 `concat(ucase(mid(name, 1, 1)), mid(name, 2)) as name`

### Hash/加密

* `md5(value)`, MD5 讯息摘要演算法运算，计算数据的 MD5 值。加盐可通过间接方p实现，比如 `md5(concat('sweetsalt', value))`
* `sha1(value)`, Security Hash Algorithm 安全散列演算法，比 MD5 值长8位 (32 v.s. 40)，但仍然可以通过碰撞攻击破解(指找到一个具有相同 hash 值的输入)
* `sha2(value, hash_length)`, SHA 2 代算法，hash 长度可选择 SHA-224, SHA-256, SHA-384, and SHA-512，即是 `hash_length` 只能为 224, 256, 384, 512 或者 0 (等同于256)。如果输入其他参数，e.g., `sha2("yellow", 222)` 则返回 `NULL`.  
* `aes_encrypt(str, key)`, AES(Advanced Encryption Standard) 加密， e.g., `select hex(aes_encrypt("password", "private_key"))；`, 其中 `hex()` 将字符串转为 16 进制, 避免加密后的输出乱码，对应的解析函数 `unhex()` 。 注：密钥的长度可任意，但官方不建议直接明文密钥 `key`，而是对其做散列演算(e.g., `sha2("key", 0)`) 之后再使用
* `aes_decrypt(str, key)`, AES 解密， e.g., `select aes_decrypt(unhex("B7646....KDKDK2", "private_key"))；`
* `compress(str)`，压缩数据，对于重复性高的数据压缩效果较好，对应解析函数的 `uncompress()`
* `statement_digist_text(state)`， 从输入字符串中提取 SQL 语句

`md5, sha1, sha2` 只支持加密不支持反向解密， 严格的来说，它们只算哈希，将无穷个数的输入转换为有穷个数的输出，在哈希演算的过程中信息有损失，函数不可逆。`md5, sha1` 通过碰撞攻击可在很短时间内找到相同输出的输入([参考资料)](https://www.cnblogs.com/baiqiantao/p/37ef1a6aa3d370ed9da0f4d336c2c646.html)。 



### 脚本模式

* 基本语法 `mysql -h host -u user -p < script.sh` 
* 控制输入 `mysql < script.sh | less (more)` or `mysql < script.sh > output.txt`
* 在 MySQL 交互界面里可通过命令 `source s.sh` or `\. s.sh` 来直接执行脚本



## 相似函数区别与联系
* `union` 与 `union all` 都会将多个结果整合出来，但存在一定的区别：`union` 会自动压缩多个结果集合中的重复结果，而 `union all` 则将所有的结果全部显示出来，不管是不是重复。

* `having` v.s., `where`, 
    * `where` 在数据聚合前过滤条件，e.g., 
    ```bash 
    select name, sum(bonus) from users where bonus bonus > 1 group by name；
    ```

    * `having` 数据聚后过滤数据， e.g., 
    ```mysql
        select name, sum(bonus) from users group by bonus having sum(bonus) > 100；  
    ```
    * 二者都定义过滤条件，`where` 对数据进行逐条过滤，而 `having` 按组过滤 .

* `group by` v.s. `partition by`, 根本区别在于是否汇总数据
    * `partition by` 没有汇总数据，用于分组，只适用于窗口函数，比如 `rank() over`； 不会影响返回的行数，但会更改窗口函数的结果计算方式； 
    * `group by` 汇总数据，对集合进行拆分； 如果有集合大小大于 1，则必然影响(减少)返回的行数 



## NULL 与三元逻辑
<img src="http://git.io/JUTft" width="400px" height="140px" alt="NULL example">

在 MySQL 中，`NULL` 不是一个值，也没有数据类型，因为它不是一个变量或者常量。它是一个表示"没有值"的标记。`NULL` 包含两种类型的含义:

1. 未知，比如空缺值
2. 不适用，比如 `1 = NULL`

通过比较谓语 `=` 来比较判断时，暗含了两条假设, 1. 等号两边都为变量或常量； 2. 类型相同。 但 `NULL` 不是数据，也没有类型，通过 `=` 比较得到的结果总是为 `NULL`，因此 `NULL = 3` 或者 `3 = NULL` 返回的结果都是 `NULL`，表示比较不适用。

如果需要判断记录里的某一项是否为空时，需要使用谓语 `v is NULL` 来处理，得到以下结果之一：
1. `1 is NULL -> false`； 
2. `NULL is NULL -> true`


### 三元逻辑

SQL-92标准

```bash
<truth value> ::= TRUE
              | FALSE
              | UNKNOWN
```

对应逻辑关系 :
* `true and unknown = unknown`,  `true or unknown = true`
* `false and unknown = false`,  `false or unknown = unknown`

在 `where` 语句中返回为 `unknown` 的记录将被过滤掉，以上表为例，如果执行 
```sql
select * from users where id > 0 and age > 1
```
将返回除第一条 id = 1 之外的所有记录。因为 `id > 1(true) and null > 0 (unknown)` 得到的值为 `unknown`.


## 效率优化
### 求 median 
Table: 'orderdetails', feature : 'unitprice', 目标求 'unitprice' 的中位值 

* 一般写法
```mysql
SELECT avg ( distinct price ) 
FROM ( 
  SELECT s1.price
  FROM sales s1, sales s2
    GROUP BY s1.price
    HAVING SUM ( CASE WHEN s2.price >= s1.price THEN 1 else 0 END ) >= COUNT (*) / 2 
    AND  SUM ( CASE WHEN s2.price <= t1.price THEN 1 else 0 END ) >= COUNT (*) / 2 ) tmp；
```
运行时间 3.97 s (MBP 15 / MySQL 8.0.12)。 复用两次表格 `t1, t2`，对两表价格一一对比，时间 `O(2 * n * n)`

* 高效率写法 
```mysql
-- After the first pass, @rownum will contain the total number of rows. 
-- This can be used to determine the median, so no second pass or join is needed.
SELECT AVG(dd.price) as median
FROM (
    SELECT price, @rownum := @rownum+1 as `row_number`, @total_rows:=@rownum
    FROM sales s, (SELECT @rownum:=0) r
        WHERE price is NOT NULL
        -- put some where clause here
        ORDER BY price
) AS dd
    WHERE dd.row_number IN ( FLOOR((@total_rows+1)/2), FLOOR((@total_rows+2)/2) )；
```
运行时间 0.01 s. 这种方法只遍历一次表，将行数存放到 `total_rows` 里面，时间复杂度 `O(n)`。 




# SQL JOINs
[C.L.Moffatt](https://link.zhihu.com/?target=https%3A//www.codeproject.com/script/Membership/View.aspx%3Fmid%3D5909363) 关于 SQL joins 的总结:
<img src='http://git.io/JUq09' width='600px'>

* `inner join` (等值联接) 只返回两个表中联结字段相等的行
* `left (outer) join` (左联接) 返回包括左表中的所有记录和右表中联结字段相等的记录
* `right (outer) join` (右联接) 返回包括右表中的所有记录和左表中联结字段相等的记录，则好与左联接对称

<img src='http://git.io/JUIAi' width='300px' alt='left/right join'>  <img src='http://git.io/JUIAX' width='300px' alt='left/right join'>

对以上两表进行左联接得到结果:
```sql 
+---------+-------+---------+----------+  
| sale_id | price | sale_id | newprice |
+---------+-------+---------+----------+
|       1 |  5000 |       1 |     1000 |
|       2 |  5000 |       2 |     3000 |
|       7 |  9000 |    NULL |     NULL |
+---------+-------+---------+----------+
```
右联接得到结果
```sql
+---------+-------+---------+----------+
| sale_id | price | sale_id | newprice |
+---------+-------+---------+----------+
|       1 |  5000 |       1 |     1000 |
|       2 |  5000 |       2 |     3000 |
|    NULL |  NULL |      10 |     1000 |
+---------+-------+---------+----------+
```

等值联接得到结果
```sql
+---------+-------+---------+----------+
| sale_id | price | sale_id | newprice |
+---------+-------+---------+----------+
|       1 |  5000 |       1 |     1000 |
|       2 |  5000 |       2 |     3000 |
+---------+-------+---------+----------+
```

## 小结
* 与 `inner join` 的区别是，`outer join` 返回的表 A, B 的交集 + (`A (left join)| B (right join) | all A + B (full join)`)
* Note: MySQL 中并没有 `full join` 命令，可以通过其他方式模拟这个命令，比如:
```sql
SELECT * FROM sa left JOIN sb WHERE sa.id = sb.id 
UNION 
SELECT * FROM sa right JOIN sb WHERE sa.id = sb.id 
```
