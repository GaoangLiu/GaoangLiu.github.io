---
layout:     post
title:      How to make life easier with Argparse 
date:       2019-10-27
tags: [Python, Script]
categories: 
- python
---

[Python3 argparse Document](https://docs.python.org/3/library/argparse.html)

I wrote lots of Python3 scripts and run it under command-line. So frequently, I found myself write something 
line 
```python
    args = sys.argv[1:]
    if args[1].endswith("@gmail"):
        do_something()    
```

This is cumbersome, ugly and hard to debug when I want to revise it.
But scripts do not have to be such ugly, if you use the `argparse` module. 

As been put in the document:
> Program defines what arguments it requires, and argparse will figure out how to parse those out of sys.argv. The argparse module also automatically generates help and usage messages and issues errors when users give the program invalid arguments.

Great, that is exactly what I need in my scripts. 

But, first, you will have to embed the module into your script by 
```python
import argparse
# and creating an object
parser = argparse.ArgumentParser()
```


# Examples and Explanations 


## Add argument 
Call the `add_argument()` method to add program argument, the calls tell the object `parser` how to take the strings on the command line and turn them into objects. E.g., 
```python
parser.add_argument("-d", "--delete", help="remove invalid account from {}".format(jsonfile))
```






