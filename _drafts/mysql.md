---
layout:     post
title:      MySQL Cheat-Sheet 
date:       2020-08-22
tags: [mysql]
categories: 
- sql
---

A simple [cheatsheet](http://git.io/JUvIJ) on MySQL. 



## 基本语法
* `delete from table [where clause]`, 按条件删除数据
* `left(str, len)`, 截取左边 `len` 长度字符。E.g., `left("abcd", 2)` = `ab`
* `substring(str, pos)`, 从 `pos` 处截取字符, e.g., `substring("abcde", 2)` = `bcde`。注意 `pos` 下标从 0 开始。如果 `pos` 为负值，则从右边开始截取
* `substring(str, pos, len)`, 截取长度为 `len` 的字符串
* `substring_index(str, delim, pos)`, 以 `delim` 为分割符，取第 `pos` 个分割符前面的内容(包含分割符)；如果 `pos` 为负值，则取倒数 `abs(pos)` 个分割符之后的内容。 E.g., `substring_index(123.344.222.222, '.', -3)` = "344.222.222".
* `mid`，是 `substring` 的别名
* `cast(value as dtype)`, 数据类型转换, e.g., `cast('2020-02-02' as date); cast(150 as char);`
* `extract(day|month|minute from date)`，从日期/时间中提取分项 
* `length(), char_length()`，分别按字节、字符形式返回字符串长度。 E.g, `select length(_utf8 '和'), char_length(_utf8 '和')` 返回 `3, 1`。对于仅由 「英文字母+数字+标点」 组成的字符串二者没有区别。
* `ucase(), lcase()` 大小写转换。E.g., 首字母大写 `concat(ucase(mid(name, 1, 1)), mid(name, 2)) as name`


### 加密
* `md5(value)`, MD5 讯息摘要演算法运算，计算数据的 MD5 值。加盐可通过间接方p实现，比如 `md5(concat('sweetsalt', value))`
* `sha1(value)`, Security Hash Algorithm 安全散列演算法，比 MD5 值长8位 (32 v.s. 40)，但仍然可以通过碰撞攻击破解
* `sha2(value, hash_length)`, SHA 2 代算法，hash 长度可选择 SHA-224, SHA-256, SHA-384, and SHA-512，即是 `hash_length` 只能为 224, 256, 384, 512 或者 0 (等同于256)。如果输入其他参数，e.g., `sha2("yellow", 222)` 则返回 `NULL`.  
* `aes_encrypt(str, key)`, AES(Advanced Encryption Standard) 加密， e.g., `select hex(aes_encrypt("password", "private_key"));`, 其中 `hex()` 将字符串转为 16 进制, 避免加密后的输出乱码，对应的解析函数 `unhex()` 。 注：密钥的长度可任意，但官方不建议直接明文密钥 `key`，而是对其做散列演算(e.g., `sha2("key", 0)`) 之后再使用
* `aes_decrypt(str, key)`, AES 解密， e.g., `select aes_decrypt(unhex("B7646....KDKDK2", "private_key"));`
* `compress(str)`，压缩数据，对于重复性高的数据压缩效果较好，对应解析函数的 `uncompress()`
* `statement_digist_text(state)`， 从输入字符串中提取 SQL 语句

`md5, sha1, sha2` 只支持加密不支持反向解密，因为在加密的过程信息有损失，函数不完全可逆。

### 脚本模式
* 基本语法 `mysql -h host -u user -p < script.sh` 
* 控制输入 `mysql < script.sh | less (more)` or `mysql < script.sh > output.txt`
* 在 MySQL 交互界面里可通过命令 `source s.sh` or `\. s.sh` 来直接执行脚本



## 关键字区分与联系
* `union` 与 `union all` 都会将多个结果整合出来，但存在一定的区别：`union` 会自动压缩多个结果集合中的重复结果，而 `union all` 则将所有的结果全部显示出来，不管是不是重复。

* `having` v.s., `where`, 
    * `where` 在数据聚合前过滤条件，e.g., 
    ```bash 
    select name, sum(bonus) from users where bonus bonus > 1 group by name;
    ```

    * `having` 数据聚后过滤数据， e.g., 
    ```mysql
        select name, sum(bonus) from users group by bonus having sum(bonus) > 100;  
    ```
    * 二者都定义过滤条件，`where` 对数据进行逐条过滤，而 `having` 按组过滤 .
