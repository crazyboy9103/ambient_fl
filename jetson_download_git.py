import paramiko
import argparse
JETSON_IP = "147.47.200.209"
MIN_PORT = "20101"
MAX_PORT = "20136"
USERNAME = "jetson"
PASSWORD = "jetson"

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
                print(f"{port} is working")
                #print(lines)

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
    for port in available:
        try:
            cli = paramiko.SSHClient()
            cli.set_missing_host_key_policy(paramiko.MissingHostKeyPolicy())
            #cli.set_missing_host_key_policy(paramiko.AutoAddPolicy)

            cli.connect(JETSON_IP, port=port, username=USERNAME, password=PASSWORD)

            stdin, stdout, stderr = cli.exec_command(command)
            

            stdin, stdout, stderr = stdin.readlines(), stdout.readlines(), stderr.readlines()
            if stdin:
                print(f"INPUT {stdin}")
            if stdout:
                print(f"OUTPUT {stdout}")
            if stderr:
                print(f"ERROR {stderr}")
            

        except Exception as e:
            print(f"Port {port} error")
            continue




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", type=str, help="linux command to execute globally")

    args = parser.parse_args()

    ping()

    command = args.command

    global_exec(command)

