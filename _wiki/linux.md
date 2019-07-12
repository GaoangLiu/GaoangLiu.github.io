---
layout: wiki
title: Linux/Unix
categories: Linux
description: 类 Unix 系统下的一些常用命令和用法。
keywords: Linux
---

类 Unix 系统下的一些常用命令和用法。

## 实用命令

### fuser

查看文件被谁占用。

```sh
fuser -u .linux.md.swp
```

### id

查看当前用户、组 id。

### lsof

查看打开的文件列表。

> An  open  file  may  be  a  regular  file,  a directory, a block special file, a character special file, an executing text reference, a library, a stream or a network file (Internet socket, NFS file or UNIX domain socket.)  A specific file or all the files in a file system may be selected by path.

#### 查看网络相关的文件占用

```sh
lsof -i
```

#### 查看端口占用

```sh
lsof -i tcp:5037
```

#### 查看某个文件被谁占用

```sh
lsof .linux.md.swp
```

#### 查看某个用户占用的文件信息

```sh
lsof -u mazhuang
```

`-u` 后面可以跟 uid 或 login name。

#### 查看某个程序占用的文件信息

```sh
lsof -c Vim
```

注意程序名区分大小写。

### crontab 

分 时 日 月 星期 要运行的命令

第 $$i$$ 列，$$i = $$

1. 分钟0～59
2. 第2列小时0～23（0表示子夜）
3. 第3列日1～31
4. 第4列月1～12
5. 第5列星期0～7（0和7表示星期天）
6. 第6列要运行的命令

```sh
# DT:delete core files,at 3.30am on 1,7,14,21,26,26 days of each month
30 3 1,7,14,21,26 * * /bin/find -name 'core' -exec rm {} \;
```

使用实例

```sh
* * * * * ls -lt /root/  # 每分钟执行一次命令 
3,15 * * * * some_cmd  # 第小时的第3， 15分钟执行一次命令
3,15 8-11 * * * some_cmd  # 在上午8点到11点的第3和第15分钟执行
3,15 8-11 */2  * * some_cmd  # 每隔两天的上午8点到11点的第3和第15分钟执行
10 1 * * 6,0 /etc/init.d/smb restart  # 每周六、周日的1 : 10重启smb

$ crontab -l # 当前crontab 任务列表
$ crontab -e # edit 
@reboot some_cmd # 开机执行
@daily some_cmd # 每天 00:00 执行 
@hourly or @weekly or @monthly or @yearly some_cmd

```

