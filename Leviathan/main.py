import socket

def scan_port (host,port,timeout = 1) : 
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s :
        s.settimeout(timeout)
        return s.connect_ex((host,port)) == 0 

host = "127.0.0.1"

for port in range(1,1026) :
    if scan_port(host,port) :
        print(f"port: {port} terbuka ")
 