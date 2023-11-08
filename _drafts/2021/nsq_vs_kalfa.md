---
layout:     post
title:      NSQ V.S. Kalfa
date:       2021-07-11
tags: [message queue, nsq, kalfa]
categories: 
- algorithm
---


# 消息队列的作用
- 解耦，将一个流程加入一层数据接口拆分成两个部分，上游专注通知，下游专注处理
- 缓冲，应对流量的突然上涨变更，消息队列有很好的缓冲削峰作用
- 异步，上游发送消息以后可以马上返回，处理工作交给下游进行
- 广播，让一个消息被多个下游进行处理
- 冗余，保存处理的消息，防止消息处理失败导致的数据丢失

# NSQ
## 组件 

NSQ主要包含3个组件：
1. nsqd：在服务端运行的守护进程，负责接收，排队，投递消息给客户端。能够独立运行，不过通常是由 nsqlookupd 实例所在集群配置的
2. nsqlookup：为守护进程，负责管理拓扑信息并提供发现服务。客户端通过查询 nsqlookupd 来发现指定话题（topic）的生产者，并且 nsqd 节点广播话题（topic）和通道（channel）信息
3. nsqadmin：一套WEB UI，用来汇集集群的实时统计，并执行不同的管理任务

## 特性
- 消息**默认不可持久化**，虽然系统支持消息持久化存储在磁盘中（通过设置 –mem-queue-size 为零），不过默认情况下消息都在内存中
- 消息最少会被投递一次(at least once)，假设成立于 nsqd 节点没有错误
- 消息无序，是由重新队列(requeues)，内存和磁盘存储的混合导致的，实际上，节点间不会共享任何信息。它是相对的简单完成疏松队列
- 支持无 SPOF 的分布式拓扑，nsqd 和 nsqadmin 有一个节点故障不会影响到整个系统的正常运行
- 支持 requeue，延迟消费机制
- 消息 push 给消费者

# Kafka
## 角色
- Producer：消息发布者，负责发布消息到Kafka broker
- Consumer：消息消费者，向Kafka broker读取消息的客户端
- Broker：Kafka集群中的一个服务器
- Topic：每条发布到Kafka集群的消息都有一个类别，这个类别被称为Topic。
- Consumer Group：Consumer Group对应NSQ的Channel，一个Consumer Group能够消费一个Topic中的所有消息。
- Partition：Parition是物理上的概念，每个Topic包含一个或多个Partition，消息保证在partition中有序，Consumer可以消费多个partition的消息。

### Topic & Partition
Topic在逻辑上可以被认为是一个queue，每条消费都必须指定它的Topic，可以简单理解为必须指明把这条消息放进哪个queue里。为了使得Kafka的吞吐率可以线性提高，物理上把Topic分成一个或多个Partition，每个Partition在物理上对应一个文件夹，该文件夹下存储这个Partition的所有消息和索引文件。

### Producer 消息路由
Producer发送消息到broker时，会根据Paritition机制选择将其存储到哪一个Partition。如果Partition机制设置合理，所有消息可以均匀分布到不同的Partition里，这样就实现了负载均衡。
在发送一条消息时，可以指定这条消息的key，Producer根据这个key和Partition机制来判断应该将这条消息发送到哪个Parition。消息在Partition中是有序的，同时一个Partition短时间内会提供给特定下游消费的Consumer 消费，这样可以提供业务中某些场景的有序保证。

### Consumer Group
使用Consumer high level API时，同一Topic的一条消息只能被同一个Consumer Group内的一个Consumer消费，但多个Consumer Group可同时消费这一消息。

这是Kafka用来实现一个Topic消息的广播（发给所有的Consumer）和单播（发给某一个Consumer）的手段。一个Topic可以对应多个Consumer Group。如果需要实现广播，只要每个Consumer有一个独立的Group就可以了。要实现单播只要所有的Consumer在同一个Group里。用Consumer Group还可以将Consumer进行自由的分组而不需要多次发送消息到不同的Topic。

## 特性
- 存储上使用了顺序访问磁盘和零拷贝技术(将磁盘文件的数据复制到页面缓存中一次，然后将数据从页面缓存直接发送到网络中)，使得其具有非常强大的吞吐性能
- 数据落磁盘，能够持久化，支持消息的重新消费
- 投递保证支持 at least one / at most one / exactly once
- Partition / Comsumer Group内保证消息有序

# 对比

## 存储
- NSQ 默认是把消息放到内存中，只有当队列里消息的数量超过–mem-queue-size配置的限制时，才会对消息进行持久化。
- Kafka 会把写到磁盘中进行持久化，并通过顺序读写磁盘来保障性能。持久化能够让Kafka做更多的事情：消息的重新消费（重置offset）；让数据更加安全，不那么容易丢失。同时Kafka还通过partition的机制，对消息做了备份，进一步增强了消息的安全性。

## 推拉模型
- NSQ 使用的是**推模型**，推模型能够使得时延非常小，消息到了马上就能够推送给下游消费，但是下游消费能够无法控制，推送过快可能导致下游过载。
- Kafka 使用的**拉模型**，拉模型能够让消费者自己掌握节奏，但是这样轮询会让整个消费的时延增加，不过消息队列本身对时延的要求不是很大，这一点影响不是很大。


## 消息的顺序性
- NSQ 因为不能够把特性消息和消费者对应起来，所以无法实现消息的有序性。
- Kafka 因为消息在Partition中写入是有序的，同时一个Partition只能够被一个Consumer消费，这样就可能实现消息在Partition中的有序。自定义写入哪个- Partition的规则能够让需要有序消费的相关消息都进入同一个Partition中被消费，这样达到”全局有序“


# References 
- [分布式消息队列 NSQ 和 Kafka 对比](https://www.liuin.cn/2018/07/11/%E5%88%86%E5%B8%83%E5%BC%8F%E6%B6%88%E6%81%AF%E9%98%9F%E5%88%97-NSQ-%E5%92%8C-Kafka-%E5%AF%B9%E6%AF%94/)
