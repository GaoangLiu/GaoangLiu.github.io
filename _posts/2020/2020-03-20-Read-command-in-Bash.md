---
layout: post
title: Read command in Bash
date: 2020-03-20
tags: linux bash shell
categories: 
author: gaoangliu
---
* content
{:toc}


**`Reading` is a good habit.**




The `read` command in the Linux can do the following things:
1. read input from the keyboard 
2. parse values from a string (work with IFS)
3. ...

## Simple read 

<img class='enter' src="{{site.baseurl}}/images/2020/simple.read.png" width="500">

`read` the data and store it into default variable `$REPLY`.

## Read with prompt text 
```bash
$ read -p "Your username: "
Your username: (wait for user to input) 
```

## Read and store value to a variable
```bash 
$ read name
John Smith
$ read -p "Your name: " name
```

## Options 
`read `
1. `-n` allows users to enter only limited length of characters
2. `-s` secure input, no echo to the terminal 
3. `-t` timeout


# Read a file
```bash
while read line; do 
        echo $line 
done < file.txt
```

## IFS
The **Internal Field Separator (IFS)** is a special shell variable, it is used for word splitting after expansion and to split lines into words with the read builtin command.

You can change the value of IFS as per your requirements.

### Combine with `read`
```bash
IFS=: read username password uid subscription <<< "John:123456:001:premium"
``` 



## Refers
1. [Read man page](http://linuxcommand.org/lc3_man_pages/readh.html)












