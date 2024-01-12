---
layout: post
title: Morris Traversal
date: 2019-12-18
tags: tree
categories: algorithm
author: GaoangLiu
---
* content
{:toc}


To traverse a given a tree $$T$$, one of most intuitive and popular way is using **stack and recursion**. 



This stack-based traversal has space complexity $$O(\text{log}N)$$, where $$N$$ is the depth of tree. 

Morris traversal is another way of traversing a tree, but requires only constant space. The idea behind it is **trading time for memory**. The thought can be found the following procedure:

1. Start from root node, mark it as `cur` node
2. While `cur` is not NULL
    - If the current node has no left child
        1. Print current node's value, `print(cur.val)`
        2. Go to the right child, `cur = cur.right`
    - Else
        1. Find the rightmost node `pre` of current node's left child, and make `cur` the right node of it, `pre.next = cur`
        2. Make `cur`'s left child as current node, `cur = cur.left`

### Python implementation, in-order traversal
```python
# Inorder tree traversal 
def MorrisTraversal(root): 
    current = root  
    while current: 
        if current.left:
            print(current.val)
            current = current.right 
        else: 
            # Find the in-order predecessor of current 
            pre = current.left 
            while pre.right and pre.right != current: 
                pre = pre.right 
   
            # Make current as right child of its in-order predecessor 
            if pre.right is None: 
                pre.right = current 
                current = current.left 
                  
            # Revert the changes made in if part to restore the  
            # original tree i.e., fix the right child of predecessor 
            else: 
                pre.right = None
                print(current.val)
                current = current.right 
```        
Notice that, the structure of the tree is modified during the traversal by the code block 
```python
if pre.right is None: 
    pre.right = current
    current = current.left
```
This block appends `cur` to the rightmost node (`pre`) of `cur`'s left child, such that we can still track back to `cur` later. Once `pre` is printed, we get `cur = pre.right = cur`. And there is a circle here when we start from `cur`, go to its left child, and then go to right all the way until we meet `cur` agin. At this time, `pre.right is not None`, then we *cut the connection* between `pre` and `cur` by running `pre.right = None`. 

In the way, the tree will be reverted back to its original shape when the algorithm ends.

### Time complexity
$$O(N)$$, as every node in the tree is traversed for at most three times. 
