#이름 변경 예정
from json_socket import Server, Message

server = Server(host='127.0.0.1', port=20000, max_con = 5)
server.accept(id = 0)
response = server.recv(id = 0)
server.send(id = 0, data="received")
print("server received", response)
