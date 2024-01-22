---
layout: post
title: Python Generators
date: 2020-04-20
tags: python generator
categories: python
author: GaoangLiu
---
* content
{:toc}


# Generators
An iterator can be seen as **a pointer to a container**, e.g. a list structure that can iterate over all the elements of this container (a list is not an iterator, but it can be used like an iterator).




Generators, on the other hand, are a special kind of function, which enable us to implement iterators.
A generator is a function which returns a generator object, which can be seen as a function that produces a sequence of results. 


## yield from 
"yield from" is available since Python 3.3!
The `yield from <expr>` statement can be used inside the body of a generator. `<expr>` has to be an expression evaluating to an iterable, from which an iterator will be extracted.

Example:
```python
length_bound = 10
def get_combinations(prev):
    if len(prev) == length_bound:
        yield prev
        return 
    for c in 'abc':
        if not prev or c != prev[-1]:
            yield from get_combinations(prev + c)
```


# References 
* [Python-Course, generators](https://www.python-course.eu/python3_generators.php)
