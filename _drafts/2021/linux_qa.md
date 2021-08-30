---
layout:     post
title:      Linux QA
date:       2021-08-29
tags: [linux]
categories: 
- linux 
---

# 扩展 swap 
```bash
swapoff -a # move stuff in swap to main memory
dd if=/dev/zero of=/swapfile bs=1M count=4096
mkswap /swapfile
chmod 600 /swapfile
swapon -a
swapon -s # check status
```

To make sure that the new swap space is activated while booting up computer, add the following line to `/etc/fstab`.
```bash
# Add this line to /etc/fstab
/swapfile swap swap sw 0 0
```