---
layout:     post
title:      Perl One-Liners
date:       2019-12-27
tags: [perl]
categories: 
- perl
---

No.1: 查看帮助 `perl -h`

## Basic
基本格式 `perl -<flags> 'codes'`，常用 flags :
- `-n`，逐行处理，等于程序中的 `while(<>){..loop..}`
- `-e`，short for 'execute'，即直接运行程序，one-line code 必备。注：`e` 必须放在所有其他 flags 的最后面，不然程序不会执行
- `-E`，在 `-e` 的基础上激活所特性，比如 `say` 
- `-l`，任意输出后添加新的一行 `perl -le 'print qq/Hello/` 等于 `perl -e 'print qq/Hello\n/'`
- `-p`，输出一行，前提 `-n` 已打开。比如 `perl -npe '' abc.txt` 将逐行输出文本

有时候输出字符串需要加上双引号，比如, `perl -e 'print "Hello one-liner"`。如果不想输入双引号(毕竟要按两次 shift)，可以使用 Perl 独有的表示方式 `q/string/` (`q` stards for quote, `qq` for double quote)。然后就可以使用 `perl -e 'print qq/Hello one-liner/` 来输出字符串了。

## Intermediated tricks
### 切割字符串

使用 `-a` 来自动分割每一行 (like `awk`)，将 `$_` 切分为数组 `@F`， 默认以空格做为分割符。如果需要指定其他分割符，比如冒号 `:` ，使用 `-F:` flag 。

```bash
cat abc.txt | perl -anE 'say $F[0]'  # = cat abc.txt | awk '{print $1}'
cat /etc/passwd | perl -al -F: -e 'say $F[2], $F[0]' 
```

### `-M` 加载模块
```bash
perl -MLWP::Simple -e 'getprint xxx' # = perl -e 'using LWP::Simple; getprint ...'
```

### 选择性输出
- 打印行数 
```bash 
perl -ne 'print "$. $_"' abc.txt # print lines preceded by line numbers, eq cat -n abc.txt
```
- 输出第一行 `perl -ne 'print; exit'`
- 输出前 10 行 `perl -ne 'print if $. <= 10' abc.txt`
- 输出最后 1 行 `perl -ne 'print if eof' abc.txt`
- 输出特定几行 `perl -ne 'print if grep $. == $_, (11,13,19)' abc.txt`。注: smartmatch `~~` 不再推荐使用
- 输出连续数行 `perl -ne 'print if 11..30' abc.txt`
- 正则输出  `perl -ne 'print if /^foo/' abc.txt`
- 正则输出并统计行数 `perl -ne 'print qq/$c $_/ if /^foo/ and ++$c' abc.tx`
- 翻转文本所有内容 `perl -e 'print reverse <>' abc.txt`
