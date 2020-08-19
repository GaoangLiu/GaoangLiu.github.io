---
layout: post
title: Docker Basics
date: 2019-09-04
tags: docker linux
categories: 
author: GaoangLau
---
* content
{:toc}


## 基本用法

### Commit

docker commit :从容器创建一个新的镜像。




`docker commit [OPTIONS] CONTAINER [REPOSITORY[:TAG]]`

* -a :提交的镜像作者；

* -c :使用Dockerfile指令来创建镜像；

* -m :提交时的说明文字；

* -p :在commit时，将容器暂停。 

`docker ps`

```bash
CONTAINER ID  IMAGE  COMMAND  CREATED  STATUS  PORTS  NAMES
d91211a28312  85c4fd36a543  "bash" 42 minutes ago Up 6 minutes  debian_slipper
```

根据这个`debian_slipper`容器提交镜像

```bash
docker commit -a "slipper0714" -m "Debian basic" d91211a28312 debian_slipper:v1
```



### Push

push镜像的规范是：
```bash
docker push [OPTIONS] NAME[:TAG]
```

推送之前，行对镜像改名，在名称前加上自己的docker hub的Docker ID，e.g., slipper0714

```bash
docker tag 61b3f03bbc49 slipper0714/debian_slipper:v1
```

然后进行推送（如果下面的v1的tag标签不写，则默认为latest）

```bash
docker push slipper0714/debian_slipper:v1  
```



## RMI

删除images

```bash
docker images
REPOSITORY            TAG                 IMAGE ID            CREATED             SIZE
debian_slipper       	v1                  61b3f03bbc49        9 minutes ago       717MB
slipper0714/debian_slipper   v1           61b3f03bbc49        9 minutes ago       717MB
ubuntu                       v1	          71e5eaf7131c        34 hours ago        862MB
slipper0714/ubuntu.v1        latest       71e5eaf7131c        34 hours ago        862MB
slipper0714docker/ubuntu     latest       a2a15febcdf3        2 weeks ago         64.2MB
debian                       latest       85c4fd36a543        3 weeks ago         114MB

```

`rmi`

```bash
docker rmi 85c4fd36a543
```





## 删除container

Spec

`docker container rm [OPTIONS] CONTAINER [CONTAINER...]`

Options:

| `--force , -f`   |      | Force the removal of a running container (uses SIGKILL) |
| ---------------- | ---- | ------------------------------------------------------- |
| `--link , -l`    |      | Remove the specified link                               |
| `--volumes , -v` |      | Remove the volumes associated with the container        |
