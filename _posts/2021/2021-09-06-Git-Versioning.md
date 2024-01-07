---
layout: post
title: Git Versioning
date: 2021-09-06
tags: git
categories: git
author: gaoangliu
---
* content
{:toc}


## 查看单个文件历史
```bash
git log --pretty=oneline file_name # 逐行显示提交信息



git log -p file_name # -p, patches 显示文件变更
git log --stat -p file_name # also show commit message
git log --fellow -p -- file_name # -p, patches 显示文件变更，包括当前文件重命名前的文件历史
git log --stat --fellow -p -- file_name # 
git log -p -2 file_name # show only the last 2 entries
```

## compare two branches
```bash
 git diff --stat --color another_branch
 ```
output example:
<img src="https://cdn.jsdelivr.net/gh/117v2/stuff@master/2021/21499dc9-f6e3-4d48-b1c6-fdc91f723a4a.png" width="800pt">


Show all file changes for a past commit.

```bash
git diff HEAD~1 
```

Show all file changes during the last 5 commits.

```bash
git diff HEAD~5
```
Append argument `--name-only` if you want to see just the name of involved files.

To show change of a certain file for a past commit
```bash
git diff a1b2c3d4^! utils/core.py
```
remove the file name if you want to see all file changes in the commit.



## `git show`
```bash
git show abc2323abc # 显示版本 abc2323 所有文件变更
```


## Give up changes on a file
```bash
git checkout c5f567 -- file1/to/restore file2/to/restore
```
This will acturally restore files to its older version. 

 