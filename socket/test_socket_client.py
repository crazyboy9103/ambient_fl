from jetson_fl import FLClient
#from json_socket import FLAGS
#import threading
import argparse

#hostname = "localhost"
#clients = [FLClient(0, host = hostname), FLClient(1, host = hostname)]
#for client in clients:
#    thread = threading.Thread(target=client.task)
#    thread.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", help="id", type=int, default=0)
    parser.add_argument("--host", help="host", type=str, default="127.0.0.1")
    parser.add_argument("--port", help="port", type=int, default=20000)

    args = parser.parse_args()
    client = FLClient(args.id, host = args.host, port = args.port)
    
    client.task()
