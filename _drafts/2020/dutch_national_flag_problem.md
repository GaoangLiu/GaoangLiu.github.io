---
layout:     post
title:      Dutch national flag problem
date:       2020-06-12
tags: [sorting, algorithm]
categories: 
---

The [Dutch national flag problem](https://en.wikipedia.org/wiki/Dutch_national_flag_problem) ([1](http://www.csse.monash.edu.au/~lloyd/tildeAlgDS/Sort/Flag/)) is a problem proposed by [Edsger Dijkstra](https://en.wikipedia.org/wiki/Edsger_Dijkstra). The flag of the Netherlands consists of three colors: *red, white and blue*. Given balls of these three colors arranged randomly in a line, the task is to arrange them such that all balls of the same color are together and their collective color groups are in the correct order.

<p align="center">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/20/Flag_of_the_Netherlands.svg/900px-Flag_of_the_Netherlands.svg.png" width='300px' alt='The Dutch national flag'>
</p>

`O(n^2)` or `O(n log n)` sorting algorithms are obvious, the real challenge of this problem is how to sort the balls in linear time, or more specifically, is there a way to solve the problem in **one-pass** using at most constant extra space ?

<p align="center">
    <img src="https://bit.ly/37VQ0jV" width='300px' alt='The Dutch national flag'>
</p>

---

The answer is to use a **three-way partitioning function** that groups items less than a given key (red), equal to the key (white) and greater than the key (blue). 

We iterate over the line of balls, and keep track of latest index of red, white and blue balls with 3 auxiliary variables, `red, white, blue`, which are initiated to `0, 0, N` (`N` the count of balls) respectively. During each iteration, if the color of current ball is:
* *red*, we swap the two balls located at `balls[red]` and `balls[white]`, and increase both variable `red, white` by 1. Consider the original color of ball `ball[red]` is:
    1. white, then we have placed two balls back to their correct order, i.e., red ball to the beginning and white ball to the middle
    2. red, we've swapped a red ball for a red ball, nothing changed
* *white*, we increase `white` by 1. White ball found in the middle, nothing needed to do here
* *blue*, similar to the first case, we swap `ball[blue]` and `ball[white]`, and decrease `blue` by one. 
    * This is the case when we find a blue ball in the middle of line, swapping balls will move it to the tail.  
    * We did not increase `white` because after the swap, the color of `ball[write]` is still unclear and requires further analysis.
     
--- 

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
