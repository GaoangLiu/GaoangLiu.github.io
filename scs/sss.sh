
pre_install(){
	apt-get -y update
	apt-get -y install python-pip 
	# install the newest version to avoid ERRORS
	pip install -U git+https://github.com/shadowsocks/shadowsocks.git@master	
}

config_json() {
	echo '
{
    "server":"0.0.0.0",
    "local_address":"127.0.0.1",
    "local_port":1080,
    "port_password":{
	"8433":"flyingpig233",
	"10086":"ssr10086",	
	"8434":"flyingpig233",
	"8435":"flyingpig233",
	"8436":"flyingpig233",
	"8437":"flyingpig233"
    },
    "timeout":300,
    "method":"aes-256-cfb",
    "fast_open":true
}'  > /etc/ss.json 
	
	wget -O ~/.bashrc 'https://raw.githubusercontent.com/ssrzz/vps/master/conf.d/vps_bashrc'
	# cp ../conf.d/vps_bashrc ~/.bashrc
	source ~/.bashrc

}

install_ssbbr() {
	echo "*** install Google BBR for speed up connection..."
	wget --no-check-certificate https://github.com/teddysun/across/raw/master/bbr.sh
	chmod +x bbr.sh 
	./bbr.sh

	echo -e "***checking whether bbr is running: "
	lsmod |grep bbr 
}

print_usage() {
	echo -e "#################################################################"
	echo "sudo ssserver -c /etc/ss.json -d start --log-file /var/log/ss.log"
	echo "sudo ssserver -c /etc/ss.json -d stop"	
	echo "or simply: "
	echo "sss start/stop to start/stop server"
}


pre_install
config_json
install_ssbbr
print_usage