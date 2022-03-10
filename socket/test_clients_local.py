from jetson_fl import FLClient
import threading

hostname = "localhost"
port = 20000
clients = []
for i in range(3):
   clients.append(FLClient(i, host = hostname, port=port))
for client in clients:
   thread = threading.Thread(target=client.task)
   thread.start()