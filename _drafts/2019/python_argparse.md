---
layout:     post
title:      Make command-line scripts easier with Argparse 
date:       2019-10-27
tags: [python, script]
categories: 
- python
---


When we execute a Python script in terminal, to pass arguments to the script, we usually adopt a strategy like: 

```python
args = sys.argv[1:]
if args[1].endswith("@gmail"):
    do_something()    
```
This is fine when the number of arguments is few, say, less than three. But when there are dozens of options to choose, codes following the above style become cumbersome, ugly and hard to debug or to refine whenever it is necessary. 
Lucky for us, with this [`argparse`](https://docs.python.org/3/library/argparse.html) module, passing and parsing arguments can be easy and elegant.

This `argparse` module is the “recommended command-line parsing module in the Python standard library” and makes it easy to write user-friendly command-line interfaces. 

`argparse` can figure out how to parse arguments out of `sys.argv` so we do not have to configure it manually. It can also automatically generates help and usage messages and issues errors when users give invalid arguments.

To start with, we need to import the library. 
```python
import argparse
# and creating an object
parser = argparse.ArgumentParser()
```

## Add argument 
Call the `add_argument()` method to add program argument, this call tells object `ArgumentParser` how to take the strings on the command line and turn them into objects. E.g., 
```python
parser.add_argument("-d", "--delete", help="remove invalid account from")
```

And by running `py prog.py`, the code returns 
```bash
usage: prog.py [-h] [-d DELETE]
optional arguments:
  -h, --help            show this help message and exit
  -d DELETE, --delete DELETE
                        remove invalid account from
```

The help message of this program contains the following information:
1. `prog.py`, the program name (regardless of where the program was invoked from)
2. A new arg `-d` we just added 

If we want to display a name other than the default `prog.py`, we can simply pass a name to the `prog=` arg. 
```python
parser = argparse.ArgumentParser('prog=simple_script')
```

### Epilog 
To display additional description, we can use the `epilog=` argument in ArgumentParser. 
```python
parser = argparse.ArgumentParser('prog=simple_script', epilog="Do go visit google.com if you want to know more")
```

### formatter_class
Classes `class argparse.RawDescriptionHelpFormatter, class argparse.RawTextHelpFormatter` help specifying an alternate formatting style. For example, to display description in multiple line: 
```python
import textwrap
parser = argparse.ArgumentParser(
     prog='PROG',
     formatter_class=argparse.RawDescriptionHelpFormatter,
     description=textwrap.dedent('''\
         Please do not mess up this text!
         --------------------------------
             1. information line 1
             2. information line 2
             ...
             998. information line 998
         '''))
```         
Produces:
<img class='center' src='https://raw.githubusercontent.com/gaoangliu/figures/master/2020/images/2019/python-argparse-1.png' width='500px'>

### add_help
By default, ArgumentParser objects add an option which simply displays the parser’s help message. For example, consider following code:
```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--foo', help='foo help')
args = parser.parse_args()
```

The type of `args` is `argparse.Namespace`. Thus, the corresponding value for argument `--foo` can be accessed with `args.foo`. 
```python
def _print(arg):
    print(arg)

if args.foo:
    _print(args.foo)
```
