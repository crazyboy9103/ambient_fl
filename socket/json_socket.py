import json, socket, pickle
from struct import pack, unpack
import numpy as np
class Message(object):
  FLAG_GET_CONFIG = 0
  FLAG_GET_ARCH = 1
  FLAG_GET_PARAMS = 2
  FLAG_GET_STATUS_CODE = 3
  FLAG_GET_DATA_IDX = 4
  FLAG_GET_DATA_NAME = 5
  FLAG_START_TRAIN = 6

  HEALTH_GOOD = 7
  HEALTH_BAD = 8
  
  def __init__(self, source, flag, data):
    self.source = source
    self.flag = flag 
    # Server -> Client 
    #        0 : compile config (optim, loss, metrics) - dict
    #        1 : model architecture - dict

    # Client -> Server
    #        2 : model parameters - list(np.array) 
    #        3 : status code - int STATUS_CODE
    #        4 : data idxs - {int client id: list(int)}
    self.data = data

  def __len__(self):
    return int(bool(self.data))

class Server(object):
  clients = {}
  def __init__(self, host, port, max_con):
    self.socket = socket.socket()
    self.socket.bind((host, port))
    self.socket.listen(max_con)

  def __del__(self):
    for i in range(len(self.clients)):
      self.close(i)

  def accept(self, i):
    if i in self.clients:
      self.clients[i]["client"].close()
    
    client, client_addr = self.socket.accept()
    self.clients[i] = {}
    self.clients[i]['client'] = client
    self.clients[i]['addr'] = client_addr
    return self

  def send(self, i, data):
    if i not in self.clients:
      self.accept(i)
    _send(self.clients[i]['client'], data)
  
  def recv(self, i):
    if i not in self.clients:
      raise Exception('Cannot receive data, no client is connected')
    
    return _recv(self.clients[i]['client'])

  def close(self, i):
    self.clients[i]['client'].close()
    del self.clients[i]

    #if self.socket:
    #  self.socket.close()
    #  self.socket = None
  def close_socket(self):
    if self.socket:
      self.socket.close()
      self.socket = None

class Client(object):
  socket = None
  id = None
  def __del__(self):
    self.close()

  def connect(self, id, host, port):
    self.id = id 
    self.socket = socket.socket()
    self.socket.connect((host, port))
    return self

  def send(self, data):
    if not self.socket:
      raise Exception('You have to connect first before sending data')
    _send(self.socket, data)
    return self

  def recv(self):
    if not self.socket:
      raise Exception('You have to connect first before receiving data')
    return _recv(self.socket)

  def recv_and_close(self):
    data = self.recv()
    self.close()
    return data

  def close(self):
    if self.socket:
      self.socket.close()
      self.socket = None


## helper functions ##
def _send(socket, data):
  data = pickle.dumps(data, protocol=3)
  data = pack('>I', len(data)) + data
  socket.sendall(data)

def _recv(socket):
  raw_msglen = recvall(socket, 4)
  if not raw_msglen:
      return None
  msglen = unpack('>I', raw_msglen)[0]
  msg =  recvall(socket, msglen)
  return pickle.loads(msg)
  
def recvall(socket, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = socket.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data