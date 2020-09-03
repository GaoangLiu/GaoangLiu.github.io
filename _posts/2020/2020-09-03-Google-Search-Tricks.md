---
layout: post
title: Google Search Tricks
date: 2020-09-03
tags: search google
categories: skill
author: GaoangLau
---
* content
{:toc}







## 精准搜索
在搜索(多个或者单个)**词条两边添加双引号**，比如 "Google 搜索技巧"，这样搜索引擎不会打乱多个词条顺序或者拆分词条。 




对比搜索结果，左图为添加双引号搜索结果，右图为未添加双引号搜索结果：

<img src="http://git.io/JUOlJ" width="380px" height='500px'> <img src="http://git.io/JUOlI" width="380px" height='500px'>


## 过滤结果 
通过在搜索词条之后添加连字符 `-word` 来排除一些结果。

适用场景：1. 语义上精确化搜索需求，有的词条对应多种结果，比如"苹果"，即对应苹果公司及其产品，也对应我们日常所食用的水果；2. 过滤搜索结果中包含的一些网址。比如某技术网站博客水平参差不齐，文章抄来抄去，在搜索过程中可以通过使用以下搜索方式: `Rabin Karp 算法 -csdn`

对比搜索结果，左图为过滤搜索结果，右图为未过滤搜索搜索结果(前 5 条搜索结果中出现 2 次)

<img src="http://git.io/JUOl4" width="380px" height="500px"> <img src="http://git.io/JUOli" height="500px" width="380px">

过滤多条结果可通过添加多个过滤词条实现，e.g., `-worda -wordb`


## site: 指定你想搜索的站点

在搜索词条之后添加 `site:stackoverflow.com` 指定只从 SO 上提取搜索结果，比如:

<img src="http://git.io/JUO8m" width="380px">


## filetype: 指定搜索文件类型
`filetype` 是 Google 开发的非常强大实用的一个搜索语法。不仅能搜索一般的文字页面，还能对某些二进制文档进行检索。支持搜索的格式包括但不限于 `.xls, .ppt, .doc, .rtf, .avi, .jpg`。 

示例: 搜索 `Rabin-Carp` 算法相关 pdf 文件 `Rabin Carp filetype:pdf`

<p style="align:center"> <img src="http://git.io/JUO82" width="380px"> </p>


## inurl: 搜索关键词在 url 链接中
`inurl` 语法返回的网页链接中包含第一个关键字，后面的关键字则出现在链接中或者网页文档中。
示例: `inurl:梵高` 注: `inurl:keyword` 之间不能有空格 


<p style="align:center"> <img src="http://git.io/JUO4J" width="380px"> </p>

类似的 `intitle` 语法指定搜索关键词包含在网页标题中。


## 使用星号通配符
即正则搜索，星号作为一个占位符，会自动匹配所有可能情况。注: 星号匹配 0 个或者多个单词。 

示例: `wake me up when * ends`

<p style="align:center">
    <img src="http://git.io/JUO48" width="380px">
</p>


## 使用 OR 搜索多个词条

示例: `iPhone OR Android`，注意这里使用的是大写的 OR (或者使用符号 `|` )，使用小写的 or 会被搜索引擎认定为用户在征求两选一相关的意见。作为对比，以下左右两图分别使用了大写与小写的 `or`:

<img src="http://git.io/JUO4P" width="380px" height="500px">  <img src="http://git.io/JUO4F" height="500px" width="380px">


## 组合使用，融会贯通

* 快速查找某人的社交档案 `周杰伦 (site:twitter.com | site:weibo.com | site:instagram.com)`
* 对比搜索水果 Apple, Lemon 并排除 Apple.com 结果 `apple | lemon -apple.com`
