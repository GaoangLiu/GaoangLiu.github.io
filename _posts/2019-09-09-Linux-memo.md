---
layout:     post
title:      Linux memo
date:       2019-09-09
img: linux4.jpg
tags: [linux]
catagories: [Linux]
---
Just memo, not a blog. 



## ffmpeg 用法 
### 添加字幕 
`ffmpeg -i input.mp4 -i input.srt -c copy -c:s mov_text output.mp4`


# 文件管理 
## Linux 挂载 WebDAV
WebDAV 的用处 [wiki](https://zh.wikipedia.org/wiki/WebDAV): 基于Web的分布式编写和版本控制（WebDAV）是超文本传输协议（HTTP）的扩展，有利于用户间协同编辑和管理存储在万维网服务器文档 。。。 

## 安装使用
Centos & Fedora & RedHat 
```bash 
yum -y install davfs2
```

Ubuntu & Debian 
```bash
apt -y install davfs2
```

安装后进行文件配置 
```bash
sed -ie 's/# use_locks       1/use_locks       0/g' /etc/davfs2/davfs2.conf
echo "Your_WebDAV_url username password" >> /etc/davfs2/secrets # eg echo "https://kita.teracloud.jp/dav/ sunnyme lovesunshine123" >> ... 
mount.davfs Your_WebDAV_url /mnt/mywebdav/
```
保存到 `secrets` 里的意义是以后再次挂载时不用重复输入名称与密码。 

### 重启自动挂载
```bash
echo "mount.davfs your_WebDAV_url /mnt/mywebdav/" >> /etc/rc.local
```


## 文件传输
### wget 
`wget` :
* `-b`, background 后台执行， `-c` 断点续传
* `-q`, quiet mode
* `-i filenames.txt`，下载`filenames.txt`中多个文件 
* `wget -r -np -nd http://example.com/packages/`， `-np --no-parent` 不遍历父目录，`-nd` 不在本机重新创建目录结构
* `--restrict-file-names=nocontrol`避免中文乱码

### youtube-dl
* `youtube-dl -f best url`, 自动下载音视频质量最好的格式
* `youtube-dl -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio' --merge-output-format mp4 url`,如果以上命令出错，可以考虑使用这条

## 内容查看与编辑
### tree
* `-L 2`, 深度设为2 

### tr -- translate characters
语法 
```bash
usage: tr [-Ccsu] string1 string2
       tr [-Ccu] -d string1
       tr [-Ccu] -s string1
       tr [-Ccu] -ds string1 string2
```
E.g., Translate to text to uppercase 
```bash
cat somefile.txt | tr [:lower:] [:upper:] # or, simply
cat somefile.txt | tr a-z A-Z
# Translate white-space to tabs
echo "How are you?" | tr [:space:] '\t'
#Translate braces into parenthesis
cat somefile.txt | tr '{}' '()'
#Squeeze repetition of characters using `-s` | 合并重复字符 
echo "Oh soooooooo             much space" | tr -s [:space:] ' '
#Delete specified characters using -d option (区分大小写)
echo "Terrible test" | tr -d 't'
#取反 `-c`，比如只保留数字
echo "char char 8742" | tr -cd [:digit:]
```

### shred

Safely overwrite and remove files. 

`shred [options] file`

* `-n`  overwrite n times, default `n=3`
* `-v` verbose 
* `-z` add final overwrite with zeros to hiding shredding
* `-u` remove file after overwritting 

Note: `rm t.txt` 并没有真正删除文件，只是删除了文件的索引，文件内容还在磁盘上。 


## 树莓派

### SSH卡顿
ssh登录后反应很慢，`ls`命令都响应半天。重启多次，还是一样结果。
与战略伙伴[Google](www.google.com)进行沟通后发现，问题可能在于路由器。于是重启路由器，然后问题解决，简直玄学。

## Apps

* Macbook network speed monitor `MenuMetersApp`





# HAProxy

 `HA` stands for `High Avaliability`. 

HAProxy提供高可用性、负载均衡以及基于TCP和HTTP应用的代理，支持虚拟主机，免费、快速并且可靠。据官方数据，其最高极限支持10G的并发。

HAProxy特别适用于那些负载特大的web站点，这些站点通常又需要会话保持或七层处理。HAProxy运行在当前的硬件上，可以支持数以万计的并发连接。并且它的运行模式使得它可以很简单安全的整合到当前的架构中，同时可以保护web服务器不被暴露到网络上。 

其支持从4层至7层的网络交换，即覆盖所有的TCP协议。

## 基本配置及说明

```bash
global               #全局设置
    log 127.0.0.1   local0   #日志输出配置，所有日志都记录在本机，通过local0输出
    #log loghost    local0 info
    maxconn 4096             #最大连接数
    chroot /usr/local/haproxy
    uid 99                   #所属运行的用户uid
    gid 99                   #所属运行的用户组
    group haproxy            #用户组
    daemon                   #后台运行haproxy
    nbproc 1                 #启动1个haproxy实例
    pidfile /usr/local/haproxy/haproxy.pid  #将所有进程PID写入pid文件
    #debug
    #quiet
 
defaults             #默认设置
    #log    global
    log     127.0.0.1       local3      #日志文件的输出定向
 
    #默认的模式:tcp|http|health
    mode   http         #所处理的类别,默认采用http模式
 
    option  httplog      #日志类别,采用http日志格式`
    option  dontlognull
    option  forwardfor   #将客户端真实ip加到HTTP Header中供后端服务器读取
    option  retries 3    #三次连接失败则认证服务器不可用
    option  httpclose    #每次请求完毕后主动关闭http通道,haproxy不支持keep-alive,只能模拟这种模式的实现
    retries 3            #3次连接失败就认为服务器不可用，主要通过后面的check检查
    option  redispatch   #当serverid对应的服务器挂掉后，强制定向到其他健康服务器
    option  abortonclose #当服务器负载很高时，自动结束掉当前队列中处理比较久的链接
    maxconn 2000         #默认最大连接数
 
    timeout connect 5000  #连接超时时间
    timeout client  50000 #客户端连接超时时间
    timeout server  50000 #服务器端连接超时时间
 
    stats   enable
    stats   uri /haproxy-stats   #haproxy监控页面的访问地址
    stats   auth test:test123    #设置监控页面的用户和密码
    stats   hide-version         #隐藏统计页面的HAproxy版本信息
 
frontend http-in              #前台
    bind    *:81
    mode    http
    option  httplog
    log     global
    default_backend htmpool   #静态服务器池
 
backend htmpool               #后台
		option allbackups  #设置了backup的时候,默认第一个backup会优先,设置option allbackups后所有备份服务器权重一样
    balance leastconn         #负载均衡算法，其他可选 source, roundrobin
    # source 根据客户端IP进行哈希的方式
    # roundrobin 轮转
    # leastcon 选择当前请求数最少的服务器
    option  httpchk HEAD /index.html HTTP/1.0    #健康检查
    server  web1 192.168.2.10:80 cookie 1 weight 5 check inter 2000 rise 2 fall 3
    server  web2 192.168.2.11:80 cookie 2 weight 3 check inter 2000 rise 2 fall 3
    # web1/web2:自定义服务器别名
    # 192.168.2.10:80:服务器IP:Port
    # cookie 1/2:表示serverid
    # weight: 服务器权重，数字越大分配到的请求数越高
    # check: 接受定时健康检查 
    # inter 2000: 检查频率
    # rise 2: 两次检测正确认为服务器可用
    # fall 3: 三次失败认为服务器不可用
 
 		#备份机器配置,正常情况下备机不会使用,当主机的全部服务器都down的时候备机会启用
    server backup1 10.1.7.114:80 check backup inter 1500 rise 3 fall 3
    server bakcup2 10.1.7.114:80 check backup inter 1500 rise 3 fall 3
 
 
listen w.gdu.me 0.0.0.0:80
    option  httpchk GET /index.html
    server  s1 192.168.2.10:80 weight 3 check
    server  s3 192.168.2.11:80 weight 3 check
 
# https的配置方法
# ------------------------------------------------------------------------------------
listen login_https_server
    bind 0.0.0.0:443   #绑定HTTPS的443端口
    mode tcp           #https必须使用tcp模式
    log global
    balance roundrobin
    option httpchk GET /member/login.jhtml HTTP/1.1\r\nHost:login.daily.taobao.net
    #回送给server的端口也必须是443
    server vm94f.sqa 192.168.212.94:443 check port 80 inter 6000 rise 3 fall 3
    server v215120.sqa 192.168.215.120:443 check port 80 inter 6000 rise 3 fall 3
 
 
# frontend配置
# --------------------------------------------------------------------------
frontend http_80_in
    bind 0.0.0.0:80   #监听端口
    mode http         #http的7层模式
    log global        #使用全局的日志配置
    option httplog    #启用http的log
    option httpclose  #每次请求完毕后主动关闭http通道,HA-Proxy不支持keep-alive模式
    option forwardfor ##如果后端服务器需要获得客户端的真实IP需要配置次参数,将可以从Http Header中获得客户端IP
 
    #HAProxy的日志记录内容配置
    capture request header Host len 40              # 请求中的主机名
    capture request header Content-Length len 10    # 请求中的内容长度
    capture request header Referer len 200          # 请求中的引用地址
    capture response header Server len 40           # 响应中的server name
    capture response header Content-Length len 10   # 响应中的内容长度(可配合option logasap使用)
    capture response header Cache-Control len 8     # 响应中的cache控制
    capture response header Location len 20         # 响应中的重定向地址
 
 
    #ACL策略规则定义
    #-------------------------------------------------
    #如果请求的域名满足正则表达式返回true(-i:忽略大小写)
    acl denali_policy hdr_reg(host) -i ^(www.gemini.taobao.net|my.gemini.taobao.net|auction1.gemini.taobao.net)$
 
    #如果请求域名满足trade.gemini.taobao.net返回true
    acl tm_policy hdr_dom(host) -i trade.gemini.taobao.net
 
    #在请求url中包含sip_apiname=,则此控制策略返回true,否则为false
    acl invalid_req url_sub -i sip_apiname=
 
    #在请求url中存在timetask作为部分地址路径,则此控制策略返回true,否则返回false
    acl timetask_req url_dir -i timetask
 
    #当请求的header中Content-length等于0时返回true
    acl missing_cl hdr_cnt(Content-length) eq 0
 
 
    #ACL策略匹配相应
    #-------------------------------------------------
    #当请求中header中Content-length等于0阻止请求返回403
    #block表示阻止请求,返回403错误
    block if missing_cl
 
    #如果不满足策略invalid_req,或者满足策略timetask_req,则阻止请求
    block if !invalid_req || timetask_req
 
    #当满足denali_policy的策略时使用denali_server的backend
    use_backend denali_server if denali_policy
 
    #当满足tm_policy的策略时使用tm_server的backend
    use_backend tm_server if tm_policy
 
    #reqisetbe关键字定义,根据定义的关键字选择backend
    reqisetbe ^Host:\ img           dynamic
    reqisetbe ^[^\ ]*\ /(img|css)/  dynamic
    reqisetbe ^[^\ ]*\ /admin/stats stats
 
    #以上都不满足的时候使用默认mms_server的backend
    default_backend mms_server
 
    #HAProxy错误页面设置
    errorfile 400 /home/admin/haproxy/errorfiles/400.http
    errorfile 403 /home/admin/haproxy/errorfiles/403.http
    errorfile 504 /home/admin/haproxy/errorfiles/504.http
```







