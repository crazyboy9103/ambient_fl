from jetson_fl import FLClient
import threading

hostname = "localhost"
clients = []
for i in range(3):
   clients.append(FLClient(i, host = hostname))
for client in clients:
   thread = threading.Thread(target=client.task)
   thread.start()