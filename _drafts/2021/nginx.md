---
layout:     post
title:      Nginx Reverse Proxy
date:       2021-09-28
tags: [nginx]
categories: 
- nginx
---


# About
Proxying is typically used to **distribute the load among several servers**, seamlessly show content from different websites, or **pass requests** for processing to application servers over protocols other than HTTP.

When NGINX proxies a request, it sends the request to a specified proxied server, fetches the response, and sends it back to the client. 
To pass a request to an HTTP proxied server, the `proxy_pass` directive is specified inside a location. For example:

```nginx
location /api/ {
    proxy_pass http://www.example.com/path/to/api/;
}
```

# A working configuration demo 

```nginx
upstream domain.cc {
  server localhost:8868;
  server 192.168.2.3:8888;
}

server {
  listen 80;
  server_name abc.cc;

  location /api {
    proxy_pass http://domain.cc; 
    proxy_redirect default;
  }


  location /hello/nginx {
    proxy_pass http://domain.cc/app;
    proxy_redirect default;
  }

  location /demo {
    proxy_pass http://domain.cc/demo;
  }

  access_log /tmp/domain_access.log;
  error_log /tmp/domain_error.log;
}
```
Noted:
1. A request for `abc.cc/api/42` will be forwarded to `localhost:8868/api/42`,
2. while a request for `abc.cc/hello/nginx/42` will be forwarded to `localhost:8868/app/42`.
Refer to [DO document](https://www.digitalocean.com/community/tutorials/understanding-nginx-http-proxying-load-balancing-buffering-and-caching) for more details.


## Elaboration
```nginx
location /api {
    proxy_pass http://domain.cc;
}
```
This will forward all requests `http://domain.cc/api/xxx` to `http://localhost:8868/api/xxx` or `http://192.168.2.3:8888/api/xxx`.

There are several **load balancing mechanisms** (or methods) are supported in nginx:
1. round-robin (default) -  requests to the application servers are distributed in a round-robin fashion,
2. least-connected — next request is assigned to the server with the least number of active connections,
3. ip-hash — a hash-function is used to determine what server should be selected for the next request (based on the client’s IP address).

```nginx
upstream awesome_app {
  # least_conn;
  # ip_hash;
  server1 localhost:9988;
  server2 localhost:9989;
}
```
With ip-hash, the client’s **IP address** is used as a hashing key to determine what server in a server group should be selected for the client’s requests. This method ensures that the requests from the same client will always be directed to the same server except when this server is unavailable.

It is also possible to forward more requests to a certain, possibly more powerful, server by using **server weights**. 
```nginx
upstream myapp1 {
    server srv1.example.com weight=3;
    server srv2.example.com;
    server srv3.example.com;
}
```
With this configuration, 60% (3/5) requests will be sent to srv1, 20% to srv2 and 20% to srv3. Server weights is **compatible with** load balancing mechanisms, i.e., server weights and the least-connected and ip-hash load balancing can be configured at the same time. 

```nginx
upstream myapp1 {
    least_conn;
    server srv1.example.com weight=3;
    server srv2.example.com;
    server srv3.example.com;
}
```


# References
- [NGINX Reverse Proxy official document](https://docs.nginx.com/nginx/admin-guide/web-server/reverse-proxy/)
- [NGINX Reverse Proxy HTTP/Socks](https://www.shuzhiduo.com/A/MyJx74bRzn/)
- [Understanding Nginx HTTP Proxying, Load Balancing, Buffering, and Caching - DigitalOcean](https://www.digitalocean.com/community/tutorials/understanding-nginx-http-proxying-load-balancing-buffering-and-caching)