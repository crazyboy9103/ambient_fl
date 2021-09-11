import paramiko
import getpass
from ping3 import ping
import time

###### JETSON NANO ADDRESS ######
JETSON_IP = "147.47.200.209"
MIN_PORT = "20101"
MAX_PORT = "20136"
EXPERIMENT = 1
MAX_ROUND = 5
TIME_DELAY = 5
USERNAME = "owner"
PASSWORD = "@mbient942"

###### SERVER ADDRESS ######
SERVER_IP = "147.47.200.178:9103"

# (Resets the server and change the total number of clients recorded in the server)
def initialize_server(client_number):
    base_url = f"http://{IP}:{PORT}/"
    client_url = base_url + "client_num"
    assert(isinstance(client_number, int))

    reset_url = base_url + "reset"
    reset_result = requests.get(reset_url)
    
    max_round_to_json = json.dumps(client_number)
    client_result = requests.put(client_url, data=max_round_to_json)
    
    assert reset_result.text == "Request OK" and client_result.text == "Request PUT OK", "Server Init Failed"
    print("Server reset success" if reset_result.text == "Request OK" else "Server reset failed")
    
max_round = 3 # Any positive integer (not tested for extremely large value)
experiment = 1 # 1,2,3,4
time_delay = 5 # time in seconds to wait until retry (not tested for extremely large value)
suppress = False # 모든 출력을 보고싶지 않으면 True
initialize_server()

#Instantiate the clients and create Threads to execute them
#client.task is recursively called within until all clients finish training (including itself)

"""
Initialize multiple Jetson Nanos and start FL
"""
class Jetson:
    def __init__(self, IP='147.47.200.22', min_port = "20101", max_port="20202"):
        assert int(min_port) < int(max_port), "max port must be >= min port"
        self.addresses = [IP+f":{i}" for i in range(int(min_port), int(max_port)+1)]
        self.available = [] #available addresses
        
        self.test_jetson_nano()
        
    def __ping(self, address):
        resp = ping(address)
        
        if resp == False:
            return False
        
        return True
    
    def __initialize(self, address, experiment, max_round, time_delay, num_samples):
        # SSH 로 접속 해서 Client들 initialize 시키고 필요한 것들 하기
        cli = paramiko.SSHClient()
        cli.set_missing_host_key_policy(paramiko.AutoAddPolicy)

        server = address  # 호스트명이나 IP 주소
        user = USERNAME
        pwd = PASSWORD

        cli.connect(server, port=22, username=user, password=pwd)
        stdin, stdout, stderr = cli.exec_command(f"python jetson_client.py --ip {SERVER_IP} --max {max_round} --delay {time_delay} --num {num_samples}")
        
        lines = stdout.readlines()
        print(''.join(lines))
        cli.close()
        
        return True
    
    
    def start_federated_learning(self):
        
    def test_jetson_nano(self):
        for address in self.addresses:
            temp = self.__ping(address)    
            if temp == True:
                self.available.append(address)
                
    def initialize_jetson_nano(self, experiment, max_round, time_delay, num_samples):
        for address in self.available:
            self.__initialize(address, experiment, max_round, time_delay, num_samples)
    
    def fetch_results(self):
        return FederatedServer.accuracies
    
    
start = time.time()
jetson = Jetson(IP=JETSON_IP, min_port = MIN_PORT, max_port=MAX_PORT)

jetson.initialize_jetson_nano(experiment=experiment, max_round=max_round, time_delay=time_delay, suppress=suppress)
jetson.start_federated_learning()

results = jetson.fetch_results()
end = time.time()

print(f"Federated learning took {end-start} seconds")

import numpy as np
import matplotlib.pyplot as plt
rounds = max(results[0].keys())
ids = max(results.keys())
result = np.zeros((ids, rounds))

for round_ in rounds:
    for cur_id in range(ids):
        result[cur_id, round_] = results[round_][cur_id]
        
def get_result(id, round):
    return result[id, round]

def plot_accuracy():
    rounds = list(range(len(result[0])))
    rounds = rounds.astype("int8")
    for fed_id in range(len(result)):
        accs = result[fed_id, :]
        plt.plot(rounds, accs, label=f"{fed_id}")
        plt.xticks(range(len(result[0]+1)))
    plt.title("Accuracies of the clients (%)")
    plt.xlabel("rounds")
    plt.ylabel("accuracy (%)")
    plt.legend()
    plt.show() 
    
plot_accuracy()
