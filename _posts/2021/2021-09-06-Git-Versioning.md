---
layout: post
title: Git Versioning
date: 2021-09-06
tags: git
categories: git
author: GaoangLau
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

## `git show`
```bash
git show abc2323abc # 显示版本 abc2323 所有文件变更
```


 