import socket
import fcntl
import struct
import netifaces as ni

#NOT USED
def get_ip_address():
	if 'eth0' in ni.interfaces():
		ni.ifaddresses('eth0')
		ip = ni.ifaddresses('eth0')[ni.AF_INET][0]['addr']
		return ip
	else:
		return 'localhost'