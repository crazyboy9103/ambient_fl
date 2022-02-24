import argparse
from client import *
import requests
import threading
import fabric
import logging, socket, paramiko.ssh_exception
from fabric import Connection, Config, SerialGroup, ThreadingGroup, exceptions, runners
from fabric.exceptions import GroupException
from random import random
import paramiko
import time


class Jetson:
    def __init__(self, min_port, max_port):
        self.address = "147.47.200.209"
        self.username, self.password = "jetson", "jetson"
        self.ports = [i for i in range(int(min_port), int(max_port)+1) if 1<=i%10<=6]
        self.ssh_ports = []
        self.connections = []
        
    def check(self):
        for port in self.ports:
            con = Connection(f'{self.username}@{self.address}:{port}', connect_kwargs ={"password":self.password})
            command = 'ls'
            print(f'----------------{port}----------------')
            try:
                con.run(command)
                self.ssh_ports.append(port)
                self.connections.append(con)
            except:
                print('ERROR')

        print("Available ports", self.ssh_ports)
        return len(self.ssh_ports)
            
    
    
    def send_command(self, command):
        for port, con in zip(self.ssh_ports, self.connections): 
            print(f'----------------{port}----------------')
            try:
                con.run(command)

            except:
                print('ERROR')

                        
    def start_fed(self, experiment, delay, max_round, num_samples, num_clients):
        for i, (port, con) in enumerate(zip(self.ssh_ports, self.connections)):
            command = f'docker exec client python3 /ambient_fl/client.py --round {max_round} --delay {delay} --num {num_samples} --id {i} --exp {experiment}'
            print(f'----------------{port}----------------')
            try:
                t=threading.Thread(target=con.run,args=(command,))
                t.start()
                time.sleep(1)
            except:
                print('ERROR')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min", required=True, default=20101, type=int)
    parser.add_argument("--max", required=True, default=20136, type=int)
    parser.add_argument("--exp", required=True, default=1, type=int)
    parser.add_argument("--round", required=True, default=5, type=int)
    parser.add_argument("--num", required=True, default=600, type=int)
    parser.add_argument("--delay", required=True, default=5, type=int)
    args = parser.parse_args()

    jetson = Jetson(min_port = args.min, max_port=args.max)
    CLIENT_NUM = jetson.check() # 통신 전에 무조건 실행되야 함

    init = requests.get(f"{args.ip}/initialize/{CLIENT_NUM}/{args.exp}/{args.round}")
    print(init)
    
    print("Kill all containers")
    jetson.send_command("docker kill $(docker ps -q)")
    print("...completed")

    print("Remove 'client' container")
    jetson.send_command("docker rm client")
    print("...completed")

    print("Pull latest image")
    jetson.send_command("docker pull crazyboy9103/jetson_fl:latest")
    print("...completed")
    print("Running the container")
    jetson.send_command("docker run -d -ti --name client --gpus all crazyboy9103/jetson_fl:latest")
    print("...completed")

    print("Starting federated learning")
    jetson.start_fed(experiment=args.exp, delay=args.delay, max_round=args.round, num_samples=args.num, num_clients=CLIENT_NUM) #important
    jetson.send_command("docker rm client")
    print("Federated learning done")
    