import paramiko
import getpass
from ping3 import ping
import time
import argparse
import requests
from tqdm import tqdm


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
                self.available.append(port)

            except:
                print(f"Port {port} Error")
                continue
                
        cli.close() 
        

    def start_fed(self, experiment, max_round, time_delay, num_samples):
        self.cli = paramiko.SSHClient()
        self.cli.set_missing_host_key_policy(paramiko.AutoAddPolicy)       
        
        for i, port in tqdm(enumerate(self.available), desc="Sending commands"):
            self.cli.connect(hostname=self.address, port=port, username=self.username, password=self.password)
            command = f"python jetson_client.py --ip {self.host_address} --max {max_round} --delay {time_delay} --num {num_samples} --id {i} --exp {experiment}"
            if port == self.available[0]:
                print(f"Sending command: {command}")
            stdin, stdout, stderr = self.cli.exec_command(command)
            if port == self.available[0]:
                lines = stdout.readlines()
                print(''.join(lines)) 
        self.cli.close()

    def init_jetson_nanos(self):
        self.check()
            
if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="Usage --ip {ip} --p {port} --max {max round} --delay {time delay} --num {num samples}")
    parser.add_argument("--minp", type=int, help="min port", default=20101)
    parser.add_argument("--maxp", type=int, help="max port", default=20106)
    parser.add_argument("--mr", type=int, help="total # of rounds to run", default=5)
    parser.add_argument("--delay", type=int, help="time delay", default=5)
    parser.add_argument("--num", type=int, help="num samples", default=600)
    parser.add_argument("--exp", type=int, help="experiment number", default=1) #2,3,4
    parser.add_argument("--serverip", type=str, help="server ip address", default="localhost:22222")
    args = parser.parse_args()
    
    ###### VARIABLES ######
    MIN_PORT = args.minp
    MAX_PORT = args.maxp
    EXPERIMENT = args.exp
    MAX_ROUND = args.mr
    TIME_DELAY = args.delay
    CLIENT_NUM = 1 + (MAX_PORT-MIN_PORT)
    assert (CLIENT_NUM > 0)
    ###### INITIALIZE SERVER ######
    import requests
    init = requests.get(f"http://localhost:22222/initialize/{CLIENT_NUM}/{EXPERIMENT}/{MAX_ROUND}")
    reset = requests.get("http://localhost:22222/reset")
    print(init, init.text)
    
    ###### SERVER ADDRESS ######
    SERVER_IP = args.serverip
    
    ###### INITIALIZE JETSONs ######
    init = requests.get(f"http://{SERVER_IP}/initialize/{CLIENT_NUM}/{EXPERIMENT}/{MAX_ROUND}")
    print(init, init.text)
    jetson = Jetson(min_port = MIN_PORT, max_port=MAX_PORT)
    jetson.init_jetson_nanos()
    jetson.start_fed(experiment=args.exp, 
                     max_round=args.mr,
                     time_delay=args.delay, 
                     num_samples=args.num)
