---
layout: post
title: Process Control
date: 2021-10-01
tags: systemd supervisord
categories: linux
author: GaoangLau
---
* content
{:toc}


# Process Control System

<img class='center' src="{{site.baseurl}}/images/2021/snes.png" width="80%">




## systemd
### Commands
- `systemctl start unit`, **start** a unit immediately
- `systemctl restart unit`, **restart** a unit immediately
- `systemctl stop unit`, **stop** a unit
- `systemctl reload unit`, **reload** a unit and its configuration
- `systemctl enable unit`, **enable** a unit to start automatically at boot
- `systemctl enable --now unit`, **enable** a unit to **start automatically** at boot and **start** it immediately
- `systemctl diable unit`, **disable** a unit to no longer start at boot
- `systemctl mask unit`, **mask** a unit to make it impossible to start


Demo
```bash
[Unit]
Description=Awesome service
After=network.target
StartLimitIntervalSec=0

[Service]
Type=simple
Restart=always
RestartSec=1
User=centos
WorkingDirectory=/path/to/project
ExecStart=/usr/bin/env python /path/to/server.py 
StandardOutput=append:/tmp/log1.log
StandardError=append:/tmp/log2.log

[Install]
WantedBy=multi-user.target
```

- The `Description` field briefes description of the current service, and the `Documentation` field gives the document location.
- `WorkingDirectory=path` setting defines on which directory the service will be launched, same as when you use cd to change a directory when you're working in the shell.
- `Restart=always`, restart the service when program failed.
- `RestartSec`, by default, systemd attempts a restart after 100ms. 
- `StandardOutput=file:path`, controls where file descriptor 1 (stdout) of the executed processes is connected to. Mode:
    - `file:path`, connect a specific file system object to standard output. 
    - `append:path` is similar to `file:path above`, but it opens the file in append mode.



## supervisord
### Demo
```bash
[program:AppName]
command=/usr/local/bin/php -f process.php 1 1 1 2           ; the program (relative uses PATH, can take args)
;process_name=%(program_name)s_%(process_num)02d ; process_name expr (default %(program_name)s)
;numprocs=3                    ; number of processes copies to start (def 1)
directory=/home/john/domains/domain/public_html/bot1                ; directory to cwd to before exec (def no cwd)
;umask=022                     ; umask for process (default None)
;priority=999                  ; the relative start priority (default 999)
autostart=false                ; start at supervisord start (default: true)
autorestart=true               ; whether/when to restart (default: unexpected)
;startsecs=1                   ; number of secs prog must stay running (def. 1)
;startretries=3                ; max # of serial start failures (default 3)
;exitcodes=0,2                 ; 'expected' exit codes for process (default 0,2)
;stopsignal=TERM               ; signal used to kill process (default TERM)
;stopwaitsecs=10               ; max num secs to wait b4 SIGKILL (default 10)
;stopasgroup=true              ; send stop signal to the UNIX process group (default false)
;killasgroup=true              ; SIGKILL the UNIX process group (def false)
user=userkdo                   ; setuid to this UNIX account to run the program
;redirect_stderr=true          ; redirect proc stderr to stdout (default false)
stdout_logfile=/tmp/app_stdout.log        ; stdout log path, NONE for none; default AUTO
stdout_logfile_maxbytes=1MB    ; max # logfile bytes b4 rotation (default 50MB)
;stdout_logfile_backups=10     ; # of stdout logfile backups (default 10)
;stdout_capture_maxbytes=1MB   ; number of bytes in 'capturemode' (default 0)
;stdout_events_enabled=false   ; emit events on stdout writes (default false)
stderr_logfile=/tmp/app_stderr.log        ; stderr log path, NONE for none; default AUTO
stderr_logfile_maxbytes=1MB    ; max # logfile bytes b4 rotation (default 50MB)
;stderr_logfile_backups=10     ; # of stderr logfile backups (default 10)
;stderr_capture_maxbytes=1MB   ; number of bytes in 'capturemode' (default 0)
;stderr_events_enabled=false   ; emit events on stderr writes (default false)
;environment=A="1",B="2"       ; process environment additions (def no adds)
;serverurl=AUTO                ; override serverurl computation (childutils)
```

- `directory`, working directory.
- `stdout_logfile`, stdout log path, NONE for none; default AUTO, max log file size.
- `stdout_logfile_maxbytes`, max logfile bytes b4 rotation (default 50MB).
- `stdout_logfile_backups`, max number of stdout logfile backups (default 10).
- `numprocs=5`, number of processes.



## Reference
- [manpage, systemd.exec.html](https://www.freedesktop.org/software/systemd/man/systemd.exec.html#StandardOutput=)
- [supervisord, official document](http://supervisord.org/)
