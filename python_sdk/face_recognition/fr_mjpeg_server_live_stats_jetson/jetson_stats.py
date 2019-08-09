
import subprocess
import sys

def jetson_gpu_usage():

	process = subprocess.Popen("cd ~/ && sudo ./tegrastats", stdout=subprocess.PIPE, shell=True)
	for line in iter(process.stdout.readline, ''):  # replace '' with b'' for Python 3
	    return int(line.split(" ")[9].split("%")[0])