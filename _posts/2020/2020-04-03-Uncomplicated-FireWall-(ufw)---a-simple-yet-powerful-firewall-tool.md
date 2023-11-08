---
layout: post
title: Uncomplicated FireWall (ufw) - a simple yet powerful firewall tool
date: 2020-04-03
tags: ufw linux
categories: linux
author: berrysleaf
---
* content
{:toc}


## Properties 
* user-friendly



* well-supported and popular in Linux community

## Configuration 
Install 

```bash
apt -y install ufw # Debain / Ubuntu 
```

Enable / Disable / Status etc . 
```bash 
sudo ufw enable / disable / reset / reload 
sudo ufw status [verbose|numbered]
```

## Examples 
By default, `ufw` deny all incoming connections, and allow all outgoing connections 

```bash
sudo ufw allow 1234 # allow incoming tcp and udp packet on port 1234
sudo ufw allow tcp/1234 # allow incoming tcp packet on port 1234, deny udp ... 

sudo ufw deny 1234 # deny incoming tcp and udp packet on port 1234

sudo ufw allow from 1.1.1.1 # allow specific IP 1.1.1.1
sudo ufw allow from 1.1.1.1 to any port 1234 # allow IP address 1.1.1.1 access to port 1234 for all protocols 
sudo ufw allow from 1.1.1.1 to any port 1234 proto tcp # allow IP address 1.1.1.1 access to port 1234 using TCP
```

Disable ping(icmp) requests, we need to edit `/etc/ufw/before.rules` and remove the following lines:
```bash 
# ok icmp codes
-A ufw-before-input -p icmp --icmp-type destination-unreachable -j ACCEPT
-A ufw-before-input -p icmp --icmp-type source-quench -j ACCEPT
-A ufw-before-input -p icmp --icmp-type time-exceeded -j ACCEPT
-A ufw-before-input -p icmp --icmp-type parameter-problem -j ACCEPT
-A ufw-before-input -p icmp --icmp-type echo-request -j ACCEPT
```
or we can simple replace all `ACCEPT` to `DROP`.

To delete rules, 
```bash 
sudo ufw status numbered 
sudo ufw delete 3 # delete the 3rd rule and rules will shift up to fill in the list.
```

To disable `IPV6` firewall rules, check out file `/etc/default/ufw` and set `IPV6=no`



