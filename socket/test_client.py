
#이름 변경 예정
from json_socket import Client, Message
client = Client()
client.connect(id=0, host='127.0.0.1', port=20000) 
client.send({"test":"test"})
response = client.recv() # blocking 
print("client received", response)
#response도 Message 인스턴스
