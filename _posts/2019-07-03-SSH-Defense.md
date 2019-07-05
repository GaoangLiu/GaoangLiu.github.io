---
layout:     post
title:      SSH Defense
subtitle:   How to keep your servers safe?
date:       2019-07-03
author:     ssrzz
catalog: 	true
tags:
  - SSH
---

`lastb`  查看失败的SSH登录尝试历史，如果一段时间内出现一连段的相同或相近的IP，那么恭喜，这台服务器被盯上了。

注： 防护工具也只是工具，不能保护服务器100%安全。 保护好服务器，首先要排除错误选项，包括但不限于： **弱密码、默认端口、不重视异常登录**.

## Endlessh

[Endlessh](https://github.com/skeeto/endlessh) is an SSH tarpit [that *very* slowly sends an endless, random SSH banner ](https://nullprogram.com/blog/2019/03/22/).

将计就计，欺骗攻击者，使其客户端(攻击工具或程序) 长时间卡住(数小时或者数日)。

### Usage

```python
Usage: endlessh [-vh] [-d MS] [-f CONFIG] [-l LEN] [-m LIMIT] [-p PORT]
  -4        Bind to IPv4 only
  -6        Bind to IPv6 only
  -d INT    Message millisecond delay [10000]
  -f        Set and load config file [/etc/endlessh/config]
  -h        Print this help message and exit
  -l INT    Maximum banner line length (3-255) [32]
  -m INT    Maximum number of clients [4096]
  -p INT    Listening port [2222]
  -v        Print diagnostics to standard output (repeatable)
```

### Tips

* 爆破手通常会选择直接尝试端口22(多数机器的默认端口，且很多用户未更改)，所以使用endless前先将端口改成一个好记但不常用的端口，比如：8964

* Endlessh 默认监听2222，为了欺骗攻击者，手动指定监听22端口 

  ```python
  endlessh -p 22
  ```

* 使用参数 `-v` 保持脚本运行日志 `endlessh -v -p 22 > end.log`



## Fail2ban

[Fail2ban](https://github.com/fail2ban/fail2ban) 基于auth日志文件工作，默认情况下它会扫描所有 auth 日志文件，如 `/var/log/auth.log`、`/var/log/apache/access.log` 等，并禁止带有恶意标志的IP，比如密码失败太多。

Fail2ban 可用于在指定的时间内拒绝 IP 地址，可以发送邮件通知。 Fail2ban能够降低错误认证尝试的速度，但是它不能消除弱认证带来的风险。 

### 安装

```ruby
git clone https://github.com/fail2ban/fail2ban.git
cd fail2ban
sudo python setup.py install , ## or 

# For Debian/Ubuntu
apt -y install fail2ban
# For CentOS 6+
yum -y install fail2ban
```



### 配置

```ruby
cp /etc/fail2ban/jail.conf /etc/fail2ban/local.conf

# emacs /etc/fail2ban/local.conf
[DEFAULT]
ignoreip = 127.0.0.1/8 192.168.1.100/24
bantime = 600
findtime = 600
maxretry = 3

# SSH servers
[sshd]
enabled = true
port = ssh
logpath = %(sshd_log)s
backend = %(sshd_backend)s
```



1. 不要直接操作主配置文件 ```jail.conf```，文件包含一组预定义的过滤器。只要有新的更新，所有配置都会被重置。
2. 参数 
   * `ignoreip` 忽略特定IP 
   * `bantime` 被禁时间(单位 second)
   * `maxtry` 是主机被禁止之前的失败次数
   * `findtime` 如果在最近 `findtime` 秒期间已经发生了 `maxretry` 次重试，则主机会被禁止 

更改配置后 `systemctl restart fail2ban.service`重启fail2ban. 

`fail2ban-client status ssh`  获取禁止的 IP ， `fail2ban-client set ssh unbanip 1.2.3.4`来释放被ban掉的IP.

