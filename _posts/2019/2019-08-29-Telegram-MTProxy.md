---
layout: post
title: Telegram-MTProxy
date: 2019-08-29
tags: telegram mtproxy
categories: 
author: GaoangLau
---
* content
{:toc}


Build [MTProxy](https://github.com/TelegramMessenger/MTProxy) for Telegram on your server (e.g., VPS), so you don't have to tolerate high-latency VPN connection. Actually, you can use Telegram without installing any VPN software on your devices. 




First, install dependencies, you would need common set of tools for building from source, and development packages for `openssl` and `zlib`.

```bash
# For Debian/Ubuntu users
apt install git curl build-essential libssl-dev zlib1g-dev 
```

## Download & Build

```bash
git clone https://github.com/TelegramMessenger/MTProxy
cd MTProxy && make 
```

The binary file will be in `objs/bin/mtproto-proxy`. Run  `make clean` if building is failed and search Google for reasons. 

## Configuration files

* Secret file: to connect to telegram servers.

  ```bash
  curl -s https://core.telegram.org/getProxySecret -o proxy-secret
  ```

* Proxy config file

  ```bash
  curl -s https://core.telegram.org/getProxyConfig -o proxy-multi.conf
  ```

You also need a generate a secret (i.e., password) to be used by users to connect to your proxy. You can always craft your onw passphrase, e.g., `d87dasfsadfa9sd8fasdf7as`, but if you want to generate a random passphrase, you can use commands like:

```bash
head -c 16 /dev/urandom | xxd -ps  # generating 16-bytes passphrase	
```

## Run MTProxy

```bash
cp objs/bin/mtproto-proxy /usr/local/bin/
/usr/local/bin/mtproto-proxy -u username -p 8888 -H 9999 -S <secret> --aes-pwd proxy-secret proxy-multi.conf -M 10
```

Where: 

* `username` is the username.
* `9999` is the port used by client devices to connect to the proxy.
* `8888` is the local port.
* `<secret>` is the password you just generated. Also you can set multiple secrets: `-S <secret1> -S <secret2>`.
* `proxy-secret` and `proxy-multi.conf` configure files curled in previous steps.
* `1` is the number of workers. You can increase the number of workers, if you have a powerful server.
* Use `mtproto-proxy --help` to read more options



## System Configuration 

1. Create systemd service file by running 

   ```bash
   emacs /etc/systemd/system/MTProxy.service # or vim /etc/systemd/system/MTProxy.service
   ```

2. Move configure files, secret, and binary proxy file to a working directory. 

   ```bash
   mkdir /opt/MTProxy/
   mv MTProxy/objs/bin/* /opt/MTProxy/
   ```

   Edit the system services

3. ```bash
   [Unit]
   Description=MTProxy
   After=network.target
   
   [Service]
   Type=simple
   WorkingDirectory=/opt/MTProxy
   ExecStart=/usr/local/bin/mtproto-proxy -u nobody -p 8888 -H 9999 -S <passphrase> --aes-pwd proxy-secret proxy-multi.conf -M 10
   Restart=on-failure
   
   [Install]
   WantedBy=multi-user.target
   ```

3. Reload daemons and restart MTProxy

4. ```bash
   systemctl daemon-reload
   systemctl restart MTProxy.service
   systemctl status MTProxy.service  # check if it's running properly
   ```

4. Autostart service after reboot

   ```bash
   systemctl enable MTProxy.service
   ```


If you see `main loop` in the output, then the program is running correctly. 

<img src="{{site.baseurl}}/images/2019/mtproxy.png" >

## Faster setup
Another git repo [mtprotoproxy](https://github.com/alexbers/mtprotoproxy/tree/stable) providing a simple, faster MTProto proxy. 

### How to use?
1. `git clone -b stable https://github.com/alexbers/mtprotoproxy.git; cd mtprotoproxy`
2. edit `config.py`, config **PORT, USERS, AD_TAG**
3. `docker-compose up -d` to run the script
4. type `docker-compose logs` to check running status and review the credentials. 

A major merit of this tool: **the proxy performance should be enough to comfortably serve about 4 000 simultaneous users on the VDS instance with 1 CPU core and 1024MB RAM.**

One thing that bugs us is we can not remove from Telegram the sponsor channel of this tool.

<img src="{{site.baseurl}}/images/2019/mtproxy-sponsor.png" width="300px" class="center">


