---
layout:     post
title:      Dutch national flag problem
date:       2020-06-12
tags: [sorting, algorithm]
categories: 
---

The [Dutch national flag problem](https://en.wikipedia.org/wiki/Dutch_national_flag_problem) ([1](http://www.csse.monash.edu.au/~lloyd/tildeAlgDS/Sort/Flag/)) is a problem proposed by [Edsger Dijkstra](https://en.wikipedia.org/wiki/Edsger_Dijkstra). The flag of the Netherlands consists of three colors: *red, white and blue*. Given balls of these three colors arranged randomly in a line, the task is to arrange them such that all balls of the same color are together and their collective color groups are in the correct order.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/20/Flag_of_the_Netherlands.svg/900px-Flag_of_the_Netherlands.svg.png" width='300px' alt='The Dutch national flag'>

`O(n^2)` or `O(log n)` sorting algorithms are obvious, the real challenge of this problem is how to sort the balls in linear time, or more specifically, how can we we solve the problem in **one-pass** and use at most constant space ?

The answer is to use a **three-way partitioning function** that groups items less than a given key (red), equal to the key (white) and greater than the key (blue). 

We iterate over the line of balls, and keep track of latest index of red, white and blue balls with 3 auxiliary variables, `red, white, blue`, which are initiated to `0, 0, N` (`N` the count of balls) respectively.
Durning the iteration, if the color of current ball is:
* *red*, we swap the two balls located at `balls[red]` and `balls[white]`, and increase both vars by 1. Consider the original color of ball `ball[red]` is:
    1. white, then we have placed two balls in their correct order, i.e., red ball to the beginning, white ball to the middle
    2. red, we've swapped a red ball for a red ball, nothing changed
* *white*, we increase `white` by 1. White ball found in the middle, nothing needed to do here
* *blue*, similar to the first case, we swap `ball[blue]` and `ball[white]`. The difference is we decrease `blue` by one and leave `white` untouched. Blue ball found in the middle, should move it the tail 


## Solution
Parameters: 
* `balls`, an array of numbers 

```python
def three_way_partition(balls):
    # Solve the Dutch national flag problem in 1 pass
    # 
    red, white, blue = 0, 0, len(balls) - 1
    while white <= blue:
        if balls[white] == 'red':
            balls[white], balls[red] = balls[red], balls[white]
            red += 1
            white += 1
        elif balls[white] == 'blue':
            balls[white], balls[blue] = balls[blue], balls[white]
            blue -= 1
        else:
            white += 1
```
