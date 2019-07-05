---
layout:     post
title:      Halting Problem
subtitle:   
date:       2019-07-05
author:     ssrzz
catalog: 	true
tags:
  - halting problem
---

## Halting Problem

停机问题: 判断任意一个程序是否能在有限的时间之内结束运行的问题。等价于如下判定问题：是否存在一个程序$$P$$，对于任意输入的程序$$\omega$$，能够判断$$\omega$$会在有限时间内结束或者死循环。

HP是不可判定的，本质上是 **一阶逻辑的不自洽性**。





## 一个简单的证明 

Source [计算机不是万能的](https://www.youtube.com/watch?v=92WHN-pAFCs)

* 假设存在一个机器 $$T$$ ，它总能判定另外一机器 $$H$$ 对于任意输入 $$\omega$$ 都能停机 
* 引入机器 $$D$$, 对于任意输入 $$\omega$$， 复制并输出两份 $$\omega$$ ; 机器 $$R$$ 读入 $$\omega \in \{ \text{stuck}, \text{not stuck} \}$$，输出 $$\neg \omega$$ 
* 拼接以上3个机器 $$D -> T -> R $$ 得到 $$X$$ 

Question: $$X$$ 输入 $$X$$ (written as $$X(X)$$) 是否停机 ？

1. 停机 ： => $$R$$ 的读入为*不停机*，=>  $$T$$ 读入 $$(X, X)$$ 输出为不停机 ，=> $$X(X)$$不停机，矛盾 ; 

2. 不停机： => $$R$$ 的读入为*停机*，=>  $$T$$ 读入 $$(X, X)$$ 输出为停机 ，=> $$X(X)$$停机，矛盾 。

   

    

