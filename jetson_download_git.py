import paramiko
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



def ping():
    for port in range(int(MIN_PORT), int(MAX_PORT)+1):
        if port % 10 > 6 or port % 10 == 0:
            continue

        cli = paramiko.SSHClient()
        #cli.set_missing_host_key_policy(paramiko.AutoAddPolicy)
        cli.set_missing_host_key_policy(paramiko.MissingHostKeyPolicy())
        try:
            cli.connect(JETSON_IP, port=port, username=USERNAME, password=PASSWORD)

            stdin, stdout, stderr = cli.exec_command("ls")
            lines = stdout.readlines()
            
            if lines:
                available.append(port)
                print(f"{port} is working: ls {lines}")

            cli.close()

        except Exception as e:
            print(f"{port} {e}")
            continue
    
    temp = [i for i in range(int(MIN_PORT), int(MAX_PORT)+1)]
    result = []

    for port in temp:
        if port % 10 <= 6 and port % 10 > 0:
            result.append(port)

    print("NOT CONNECTED PORTS : ", list(set(result)-set(available)))


        

def global_exec(command):
    # command : linux command to execute 
    print(f"Executes '{command}'")
    for port in available:
        try:
            cli = paramiko.SSHClient()
            cli.set_missing_host_key_policy(paramiko.MissingHostKeyPolicy())
            #cli.set_missing_host_key_policy(paramiko.AutoAddPolicy)

            cli.connect(JETSON_IP, port=port, username=USERNAME, password=PASSWORD)
            
            _, stdout, stderr = cli.exec_command(f"{command}")
            
            if stdout:
                stdout = stdout.readlines()
            if stderr:
                stderr = stderr.readlines()
            
            if stdout:
                print(f"Port {port} OUTPUT {stdout}")
            if stderr:
                print(f"Port {port} ERROR {stderr}")
            

        except Exception as e:
            print(f"Port {port} error {i}")
            continue


if __name__ == "__main__":
    ping()
    global_exec(COMMAND)
