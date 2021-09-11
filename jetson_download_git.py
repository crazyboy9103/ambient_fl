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
        cli = paramiko.SSHClient()
        cli.set_missing_host_key_policy(paramiko.AutoAddPolicy)

        try:
            cli.connect(JETSON_IP, port=port, username=USERNAME, password=PASSWORD)

            stdin, stdout, stderr = cli.exec_command(f"ls")
            lines = stdout.readlines()
            
            if lines:
                available.append(port)
            cli.close()

        except Exception as e:
            continue

    print("NOT CONNECTED PORTS : ", set([i for i in range(int(MIN_PORT), int(MAX_PORT)+1)])-set(available))


        

def global_exec(command):
    # command : linux command to execute 
    for port in available:
        try:
            cli = paramiko.SSHClient()
            cli.set_missing_host_key_policy(paramiko.AutoAddPolicy)

            cli.connect(JETSON_IP, port=port, username=USERNAME, password=PASSWORD)

            stdin, stdout, stderr = cli.exec_command(f"{command}")
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

