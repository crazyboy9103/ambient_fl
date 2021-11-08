import paramiko
import getpass
from ping3 import ping
import time
import requests


class Jetson:
    def __init__(self, min_port, max_port):
        assert int(min_port) < int(max_port), "max port must be >= min port"
        self.address = "147.47.200.209"
        self.host_address = SERVER_IP
        self.username, self.password = "jetson", "jetson"
        self.ports = [i for i in range(int(min_port), int(max_port)+1)]        
        self.available = []
        
    def check(self):
        # check which clients are online 
        cli = paramiko.SSHClient()
        cli.set_missing_host_key_policy(paramiko.AutoAddPolicy)       
        
        for port in self.ports:
            try:
                cli.connect(hostname=self.address, port=port, username=self.username, password=self.password)
                stdin, stdout, stderr = cli.exec_command("ls")
                lines = stdout.readlines()
                print(''.join(lines))
                self.available.append(port)
            except Error as e:
                print(f"Port {port} Error")
                continue
                
        cli.close() 
        

    def start_fed(self, experiment, max_round, time_delay, num_samples):
        self.cli = paramiko.SSHClient()
        self.cli.set_missing_host_key_policy(paramiko.AutoAddPolicy)       
        
        for i, port in enumerate(self.available):
            cli.connect(hostname=self.address, port=port, username=self.username, password=self.password)
            command = f"python jetson_client.py --ip {self.host_address} --max {max_round} --delay {time_delay} --num {num_samples} --id {i} --exp {experiment}"
            if port == self.available[0]:
                print(f"Sending command: {command}")
            stdin, stdout, stderr = cli.exec_command(command)
            if port == self.available[0]:
                lines = stdout.readlines()
                print(''.join(lines)) 
        self.cli.close()

    def init_jetson_nanos(self):
        self.check()
            
if __name__ == "__main__":
    ###### VARIABLES ######
    MIN_PORT = "20101"
    MAX_PORT = "20136"
    EXPERIMENT = 1
    MAX_ROUND = 5
    TIME_DELAY = 5
    CLIENT_NUM = 5

    ###### SERVER ADDRESS ######
    SERVER_IP = "147.47.200.178:22222"
    ###### Initialize ######
    init = requests.get(f"http://{SERVER_IP}/initialize/{CLIENT_NUM}/{EXPERIMENT}/{MAX_ROUND}")
    print(init, init.text)
    
    jetson = Jetson(min_port = MIN_PORT, max_port=MAX_PORT)
    jetson.init_jetson_nanos(experiment=experiment, max_round=max_round, time_delay=time_delay, suppress=suppress)
    jetson.start_fed()
    print(f"Federated learning experiment:{EXPERIMENT}, max round: {MAX_ROUND}, time delay: {TIME_DELAY}, client num: {CLIENT_NUM}, took {end-start} seconds")
