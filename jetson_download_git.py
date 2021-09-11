import paramiko
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--command", type=str, default="ls", help="linux command to execute globally")
parser.add_argument("--min", type=str, default="20101")
parser.add_argument("--max", type=str, default="20136")
args = parser.parse_args()
JETSON_IP = "147.47.200.209"
MIN_PORT = args.min
MAX_PORT = args.max
USERNAME = "jetson"
PASSWORD = "jetson"


COMMAND = args.command


   

available = []


class GlobalExecute:
    def __init__(self):
        self.cli = paramiko.SSHClient()
        self.cli.set_missing_host_key_policy(paramiko.MissingHostKeyPolicy())
        
        self.clients = {}
        self.ping()

        pass

    def ping(self):
        self.available = []
        for port in range(int(MIN_PORT), int(MAX_PORT)+1):
            if port % 10 > 6 or port % 10 == 0:
                continue

            try:
                self.available.append(port)

                cli = paramiko.SSHClient()
                cli.set_missing_host_key_policy(paramiko.MissingHostKeyPolicy())
                cli.connect(JETSON_IP, port=port, username=USERNAME, password=PASSWORD)

                stdin, stdout, stderr = cli.exec_command("ls")
                lines = stdout.readlines()
            
                if lines:
                    self.clients[port] = cli
                    print(f"{port} is working: ls {lines}")

                
            except Exception as e:
                print(f"{port} {e}")
                continue
    
        temp = [i for i in range(int(MIN_PORT), int(MAX_PORT)+1)]
        result = []

        for port in temp:
            if port % 10 <= 6 and port % 10 > 0:
                result.append(port)

        print("NOT CONNECTED PORTS : ", list(set(result)-set(self.available)))

    def global_exec(self, command):
        # command : linux command to execute 
        print(f"Executes '{command}'")
        for port in self.available:
            try:
                cli = self.clients[port]
                stdin, stdout, stderr = cli.exec_command(f"{command}")
            
                if stdout:
                    stdout = stdout.readlines()
                if stderr:
                    stderr = stderr.readlines()
            
                print(f"Port {port} OUTPUT {stdout}")
                
                if stderr:
                    print(f"Port {port} ERROR {stderr}")
            

            except Exception as e:
                print(f"Port {port} error {e}")
                continue


    
if __name__ == "__main__":
    cl = GlobalExecute()
    cl.global_exec(COMMAND)
    
    res = "y"
    while res == "y":
        res = input("More commands? (y/n)\n")
        if res == "y":
            new_command = input("new command:\n")
            new_command.strip("\n")
            cl.global_exec(str(new_command))
        
        else:
            for k, v in cl.clients.items():
                v.close()
            sys.exit()


            
