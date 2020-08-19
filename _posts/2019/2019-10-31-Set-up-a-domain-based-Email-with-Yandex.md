---
layout: post
title: Set up a domain-based Email with Yandex
date: 2019-10-31
tags: mail yandex
categories: note
author: GaoangLau
---
* content
{:toc}


搭建个人域名邮箱有很多好处，自定义任意前缀，无需繁琐的注册过程，更不用担心注册信息泄露而成为大数据中的一员。




域名邮箱搭建有多种方式，我们推荐使用[Yandex](https://www.yandex.com)(俄罗斯互联网巨头，旗下搜索引擎在本土市场占有率超60%)的服务来完成这项任务。推荐原因包括但不限于：

- 注册相比自行搭建步骤简单
- 由Yandex做担保，稳定可靠，无需自行维护，且比一般小众邮箱更为安全
- 可自定义 1000 个邮箱(前缀)，每个用户10G容量，多开邮箱账号无需门槛

一言以蔽之，快速、方便、省心。 

## Prerequisites 

1. 个人域名 (E.g., [name.com](https://www.name.com/referral/388121)上$1.99的`xyz`域名)
2. Yandex 邮箱 (注册入口 [mail.yandex.com](https://mail.yandex.com))

假设个人域名为 `flyingkiwi.me`。

## 流程
### 1. 注册Yandex Mail

注册入口 [passport.yandex.com](https://passport.yandex.com/registration)。注册过程中会要求提供个人手机号码，如果不想提供，也可以通过设置安全问题来完成注册。 

注：刚注册完的邮箱在24小时内尽量不要发送邮件，有可能被Yandex认定为spammer. 

<img class='center' src="{{site.baseurl}}/images/2019/passport.yandex.png" width="80%">

### 2. 注册域名邮箱 
个人域名邮箱申请入口 [Connect.Yandex](https://connect.yandex.com/pdd/)。
<img class='center' src="{{site.baseurl}}/images/2019/connect.yandex.png" width="500px">

填入 `flyingkiwi.me`进行注册。之后需要确认域名所有权，并将域名委托至 Yandex 服务器。

确认所有权主要通过以下4种方式 ：

1. Upload a file to the site directory
2. Add a CNAME record
3. Change the contact address through the registrar
4. Delegate a domain to Yandex servers

很多人推荐第3种方法，即设置DNS record，即在DNS控制面板处添加TXT记录
```bash
TXT yandex-verification 87cal27lacla22c2
```

但实际上第一种方式更快速一些，即直接在服务器(VPS)根目录创建相应文件([visit for more details](https://yandex.com/support/domain/setting/confirm.html#way1))。基本上在3分钟之内就能被Yandex验证成功。 

### 3. 配置MX, SPF and DKIM 
Question : Why ?

Answer: 邮件服务器解析、降低个人域名邮件被标记为垃圾邮件的风险。 

#### 配置MX
在DNS管理平台上添加 MX 记录，优先级填写 10 
<img class='center' src="{{site.baseurl}}/images/2019/mx.yandex.png" width="80%">

#### SPF, DKIM

在DNS管理平台上添加 TXT 记录 ([VISIT for more details](https://yandex.com/support/domain/set-mail/spf.html)). 

`Name: @  Value: v = spf1 redirect = _spf.yandex.net`

DKIM signature is used to verify that whether a message really came from the supposed sender.
To do so, we need to create a TXT record for the domain `flyingkiwi.me` with a public key signature, which can be generated from [HERE](https://connect.yandex.com/portal/admin/customization/mail). 
For more detail, please visit [THIS PAGE](https://yandex.com/support/domain/set-mail/dkim.html).

If all the previous steps are strictly followed, we should by now have our own domain-based Mail system. 
Add a new user is simple, just visit [PROTAL.Yandex](https://connect.yandex.com/portal/admin/departments/1) and hit `add a person`, the rest is straightforward. 

## Config domain-based email on iPhone
`Passwords & Accounts` ➡ `Add Account` ➡ `Other` ➡ `Add Mail Account`. 

And then type in `i@flyingkiwi.me`(your first domain-based email) and password and go next. Other infos and configurations are shown in the following image. 

Note that, the port for SMTP server `smtp.yandex.com` is 465, while the default port given by iPhone is 587.

<img class='center' src="{{site.baseurl}}/images/2019/iphone.yandex.png" width="50%">

```
POP3: pop.yandex.com 995
SMTP smtp.yandex.com 465
IMAP imap.yandex.com 993
```

## A DNS configuration example

<img class='center' src="{{site.baseurl}}/images/2019/summary.yandex.png" width="100%">
