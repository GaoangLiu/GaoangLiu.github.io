---
layout:     post
title:      Python Dataclasses
date:       2021-10-15
tags: [dataclass]
categories: 
- python
---

<img src="https://cdn.jsdelivr.net/gh/ddots/stuff@master/2021/work_in_progress_pixabay.png" width="50%">

A data class is a class typically containing mainly data. It is created using the `@dataclass` decorator from `dataclasses` module. The decorator `@dataclass` automatically adds generated special methods such as `__init__()` and `__repr__()` to user-defined classes. 

A simple example:   
```python
@dataclass
class User:
    name: str
    age: int = 10
```
Rules: 
- A field without a default vallue must occur before fileds with default values.


# Decorator parameters
By default, a simple `@detaclass` (without signature and brackets) is equivalent to the following usage:
```python
@dataclasses.dataclass(*, init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, match_args=True, kw_only=False, slots=False)
```
The parameters to `dataclass` are:
- `init`, default `True`. A `__init__()` method will be generated. 
- `repr`, default `True`. A `__repr__()` method will be generated. The generated repr string will have the *class name* and *the name and repr of each field*, in the order they are defined in the class. Fields marked as excluded, e.g., with type `typing.ClassVar`, from the repr will not be included.
- `eq`, default `True`. An `__eq__()` method will be generated. This method compares the class as if it were a tuple of its fields, in order. Both instances in the comparison must be of the identical type. `eq` must be `True` when `order` is `True`, otherwise, `ValueError` is raised (as shown in the following image). 
- `order`, default `False`. If `True`, `__lt__()`, `__le__()`, `__gt__()`, and `__ge__()` methods will be generated. 
- `unsafe_hash`, default `False`. A `__hash__()` method is generated according to how eq and frozen are set.
- `frozen`, default `False`. If `True`, assigning to fields will generate an exception. This emulates read-only frozen instances.
- `match_args`, default `True`
- `kw_only`, default `False`. If `True`, then all fields will be marked as keyword-only. 
- `slots`, default `False`. 

The later three parameters `match_args`, `kw_only`, `slots` are new in Python 3.10. 



<img src="https://cdn.jsdelivr.net/gh/ddots/stuff@master/2021/fc1c74f8-4935-4d2f-9cca-c84677f5f6b3.png" width="70%">


# Customize Python dataclass fields with the `field` function
The default way dataclasses work should work for most use cases.
- In some scenarios, you may need to create an attribute that is defined only internally, rather than when the class is instantiated. This may be the case when the value of the attribute depends on a previously set attribute.
This can be achieved with `field` function

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class User:
    name:str
    score: int = field(compare=False)    
    weight: float = field(default=0.0, repr=False)
    age: int = 0
    tasks: List[str] = field(default_factory=list)
```
## Parameter `default` v.s. `default_factory`
- `default`, if provided, will be the **immutable** default value for the field, e.g., `weight: float = field(default=0.0)`. 
    - Immutable means this paramaters work on data types such as **int, float, decimal, bool, string, tuple, and range**. If user assign a `dict` to a field with `default` parameter, for example,  `info:Dict[int, int] = field(default={1:42})`, `ValueError` will be raised.

- `default_factory`, requires a **zero-argument callable** that will be called when a default value is needed for a field. This parameter can be used to specify fields with **mutable** default values, values of type list, dict, set or user-defined classes.


# Use `__post_init__` to control Python dataclass initialization
If `__post_init__()` is defined on the class, the generated `__init__()` code will call a method named `__post_init__()`. This allows for initializing field values that depend on one or more other fields. For example,  

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class User:
    name: str
    score: int = field(compare=False)    
    weight: float = field(default=0.0, repr=False)
    age: int = 0
    tasks: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.basic_info = [self.name, self.age]
```

# Use `InitVar` to control Python dataclass initialization
If a field is an `InitVar`, it is considered a pseudo-field called an init-only field. As it is **NOT a true field**, it is not returned by the module-level `fields()` function, or `dataclasses.asdict()`, `dataclasses.astuple()`. 

One use of `InitVar` is to customize Python dataclass setup. It allows you specify a pseudo-field that will be passed to `__init__` and then to `__post_init__` so that you can complete certain tasks, possibly do some filtering of a `List` type field when certain condition was fullfilled, on dataclass initialization.

For example, only assign tasks to users in 'Good' `condition`:

```python
from dataclasses import dataclass, field, InitVar
from typing import List

@dataclass
class User:
    name: str
    score: int = field(compare=False)    
    weight: float = field(default=0.0, repr=False)
    age: int = 0
    tasks: List[str] = field(default_factory=list)
    condition: InitVar[str] = None 
    
    def __post_init__(self):
        if condition == 'Good':
            self.tasks = ['read', 'code']
```

# Methods
## `dataclasses.asdict`
`dataclasses.asdict(instance, *, dict_factory=dict)`

Converts the dataclass instance to a dict, using the factory function `dict_factory`. Each dataclass is converted to a dict of its fields, as `name: value` pairs. dataclasses, dicts, lists, and tuples are recursed into. For example:
```python
@dataclass
class User:
    name: str
    extra: typing.Set[str] = field(
        default_factory=lambda: ({"interests", "devices"}))
    info: typing.Dict[int, int] = field(default_factory=lambda: ({
        1: 42,
        2: 99
    }))

u = User('john')
r = dataclasses.asdict(u) # {'name': 'john', 'extra': {'devices', 'interests'}, 'info': {1: 42, 2: 99}}
```

## `astuple`
`dataclasses.astuple(instance, *, tuple_factory=tuple)`

Converts the dataclass instance to a tuple, using the factory function `dict_factory`. Each dataclass is converted to a tuple of its fields, as `name: value` pairs. dataclasses, dicts, lists, and tuples are recursed into. Note that, only values are retained, while the keys of fields are discarded.  

## `replace`
`dataclasses.replace(instance, /, **changes)`
Creates a new object of the same type as instance, replacing fields with values from `changes`. 
```python

@dataclass
class User:
    name: str
    extra: typing.Set[str] = field(
        default_factory=lambda: ({"interests", "devices"}))
    info: typing.Dict[int, int] = field(default_factory=lambda: ({
        1: 42,
        2: 99
    }))
u = User('john')
print(dataclasses.replace(u, extra=(42), name='toy', info=23)) # User(name='toy', extra=42, info=23)
```

Note that, there is a bug([StackOverflow: Python dataclasses.replace not working for InitVar](https://stackoverflow.com/questions/59597283/python-dataclasses-replace-not-working-for-initvar)) on `replace()` for Python 3.7, 3.8, 3.9.  If init-only fileds are presented in the data class, you must munally pass then to `replace()` so that they can be passed to `__init__()` and `__post_init__()`, even if you have no desire to change their values. Otherwise, your codes will respond with `ValueError`:

> ValueError: InitVar 'int_value' must be specified with replace()

This bug was [fixed](https://github.com/python/cpython/commit/bdee2a389e4b10e1c0ab65bbd4fd03defe7b2837) in Python 3.10.

## `is_dataclass`
Return True if its parameter is a `dataclass` or an instance of one, otherwise return False.

`dataclasses.is_dataclass(instance_or_class)`

## `make_dataclass`
`dataclasses.make_dataclass(cls_name, fields, *, bases=(), namespace=None, init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, match_args=True, kw_only=False, slots=False)`  

Creates a new dataclass with name `cls_name`, fields as defined in `fields`, base classes as given in `bases`, and initialized with a namespace as given in namespace. 
`fields` is an iterable whose elements are each either `name`, `(name, type)`, or `(name, type, Field)`.

```python
Coda = dataclasses.make_dataclass(
    'Coda', [('name', str, 'Cola'), ('year', int, 3039)],
    namespace={'add_one': lambda self: self.year + 1})
```

# When to use Python dataclasses — and when not to use them
- One common scenario for using `dataclasses` is as a replacement for the `namedtuple`. Dataclasses offer the same behaviors and more, and they can be made immutable (as `namedtuples` are) by simply using `@dataclass(frozen=True)` as the decorator.
- Another possible use case is replacing nested dictionaries, which can be clumsy to work with, with nested instances of `dataclasses`. 

Not every Python class needs to be a dataclass. If the purpose of a class is not a container for data, you don't have to make it a dataclass.


# References
- [How to use Python dataclasses](https://www.infoworld.com/article/3563878/how-to-use-python-dataclasses.html)
