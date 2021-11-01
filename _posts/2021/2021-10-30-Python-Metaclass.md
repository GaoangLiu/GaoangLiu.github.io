---
layout: post
title: Python Metaclass
date: 2021-10-30
tags: metaclass
categories: python
author: GaoangLau
---
* content
{:toc}




The type of an object is called a class, and the class of a class is called a **metaclass**.




Create a class with `type`:
> `type(class_name, tuple_of_parent_class, dict_of_attribute_names_and_values)`

where, 
- `name`: name of the class.
- `bases`: tuple of the parent class (for inheritance, can be empty).
- `attrs`: dictionary containing attributes names and values. `type` accepts a dictionary to define the attributes of the class.
E.g., `type("Hello", (), {})`


# Usage
In most cases, there is no need for users to master the concept of metaclass or implement in codes. Nevertheless, we present here a few examples to illustrate how to use Python metaclass.

## Case 1: force subclasses to implement specific methods
```python
class Meta(type):
    def __new__(cls, name, bases, namespace, **kwargs):
        if name != 'Base' and 'bar' not in namespace:
            raise TypeError('bad user class')
        return super().__new__(cls, name, bases, namespace, **kwargs)

class Base(object, metaclass=Meta):
    def foo(self):
        return self.bar()

class Derived(Base):
    ...
```
Returns:
```bash
Traceback (most recent call last):
  File "pymeta.py", line 48, in <module>
    class Derived(Base):
  File "pymeta.py", line 39, in __new__
    raise TypeError('bad user class')
TypeError: bad user class
```

## Case 2: register subclasses
```python
class Meta(type):
    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        if not hasattr(cls, 'registory'):
            # this is the base class
            cls.registory = {}
        else:
            # this is the subclass
            cls.registory[name.lower()] = cls

class Fruit(object, metaclass=Meta):
    pass

class Apple(Fruit):
    pass

class Orange(Fruit):
    pass

Fruit.registory 
# {'apple': <class '__main__.Apple'>, 'orange': <class '__main__.Orange'>}
```



# References
- [Understanding Python Metaclass](https://lotabout.me/2018/Understanding-Python-MetaClass/)
- [What are metaclass in Python - stackoverflow](https://stackoverflow.com/questions/100003/what-are-metaclasses-in-python)