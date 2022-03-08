from jetson_fl import FLClient
from json_socket import FLAGS
import threading

hostname = "localhost"
clients = [FLClient(0, host = hostname), FLClient(1, host = hostname)]
for client in clients:
   thread = threading.Thread(target=client.task)
   thread.start()