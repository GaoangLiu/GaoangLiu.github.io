---
layout:     post
title:      Add ipv6 tunnel to VPS
date:       2020-03-17
img: linux4.jpg
tags: [linux, vps, ipv6]
catagories: [Linux]
---

## Why
With an ipv6 address, you can visit `ipv6.google.com` and PT websites such as `bt.byr.cn`; 

## How
#1 Login [tunnelbroker](https://tunnelbroker.net/) (register if you don't have an account) 


#2 Create a regular tunnel 

<img class='center' src="{{site.baseurl}}/images/2020/add_tunnel.png" width="350">

#3 Type in the ipv4 address of your VPS 

<img class='center' src="{{site.baseurl}}/images/2020/ipv4_endpoint.png" width="350">

#4 Add configurations to `/etc/network/interfaces`, this depends on you OS version, e.g., for `Debian/Ubuntu`, paste the following lines to `/etc/network/interfaces` will do the job.

```bash
auto he-ipv6
iface he-ipv6 inet6 v4tunnel
        address 2001:470:23:5e5::2
        netmask 64
        endpoint 74.82.46.6
        local 18.180.20.104
        ttl 255
        gateway 2001:470:23:5e5::1
```

#5 Turn on service by running `ifup he-ipv6`, or restart your system to make sure things work out. 

## F.Y.I
At most 5 tunnels can be created for each account. 










