from jetson_fl import FLClient
from json_socket import FLAGS
import threading


clients = [FLClient(0), FLClient(1)]
for client in clients:
    thread = threading.Thread(target=client.task)
    thread.start()
    