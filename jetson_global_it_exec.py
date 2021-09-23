import paramiko
import time
import threading
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--min", type=int, default=20101, help="minimum port number")
parser.add_argument("--max", type=int, default=20131, help="maximum port number")
parser.add_argument("--client", type=int, default=5, help="total number of clients")
parser.add_argument("--experiment", type=int, default=1, help="(1,2,3,4) experiment")
args = parser.parse_args()


class JetsonSSHClient:
  def __init__(self, port):
    self.port = port
    self.cli = paramiko.SSHClient()
    self.cli.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    self.cli.connect("147.47.200.209", port=port, username="jetson", password="jetson")
    self.channel = self.cli.invoke_shell()
    
  def execute(self, command = "ls -al"):
    command = command +"\n"
    self.channel.send(command)
    # 결과 수신
    
    time.sleep(3)
    output = self.channel.recv(65535).decode("utf-8")

    if self.port == 20101:
      print(output.strip(command))
    #print(output.split("jetson@jetson-desktop:~$")[0])
    #print(output.split("jetson@jetson-desktop:~$")[1])

  def close(self):
    self.cli.close()

class JetsonController:
  def __init__(self, min_port = 20101, max_port = 20101):
    self.clients = {}
    self.threads = {}
    for port in range(min_port, max_port + 1):
      self.clients[port] = JetsonSSHClient(port)
      self.threads[port] = None

  def global_execute(self, command = "ls -al"):
    for port, client in self.clients.items():
      thread = threading.Thread(target = client.execute, args=(command,))
      self.threads[port] = thread
      thread.start()
      #client.execute(command)

  def check_status(self):
    for thread in self.threads.values():
      if thread.is_alive() == True:
        return False
    return True







if __name__ == "__main__":
  

  import requests
  import json
  client_num_in_json = json.dumps(args.client)
  res = requests.put("147.47.200.178:9103/client_num", data=client_num_in_json)
  print("client number put", res.text)
  exp_in_json = json.dumps(args.experiment)
  res = requests.put("147.47.200.178:9103/experiment", data=exp_in_json)
  print("experiment put", res.text)

  jetsonController = JetsonController(min_port=args.min, max_port=args.max)
  command = "ls -al"
  jetsonController.global_execute(command)
  


  while True:
    command = input()
    jetsonController.global_execute(str(command))
