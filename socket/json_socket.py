import json, socket, pickle
from struct import pack, unpack
import numpy as np
class Message(object):
  FLAG_GET_OPTIM = 0
  FLAG_GET_LOSS = 0
  FLAG_GET_METRICS = 0
  FLAG_GET_ARCH = 1
  FLAG_GET_CURRENT_ROUND = 2
  FLAG_GET_MAX_ROUND = 2
  FLAG_GET_PARAMS = 3
  FLAG_GET_ERROR_CODE = 4
  FLAG_GET_NUM_DATA = 5

  def __init__(self, source, flag, data):
    self.id = source
    self.flag = flag 
    # Server -> Client 
    #        0 : compile config (optim, loss, metrics) - dict
    #        1 : model architecture - dict
    #        2 : current round / max round - dict 

    # Client -> Server
    #        3 : model parameters - list(np.array) 
    #        4 : error - int ERROR_CODE
    #        5 : num data - int num_data
    self.data = data

  def get_id(self):
    return self.id

  def get_optim(self):
    # flag 0
    return self.data["optim"]
  
  def get_loss(self):
    # flag 0
    return self.data["loss"]
  
  def get_metrics(self):
    # flag 0
    return self.data["metrics"]

  def put_config(self, config):
    self.data = config # contains optim, loss, metrics
  
  def get_arch(self):
    # flag 1
    return self.data
  def put_arch(self, arch):
    self.data = arch

  def get_current_round(self):
    # flag 2
    return self.data["curr_round"]

  def put_round(self, current_round, max_round):
    self.data = {"curr_round":current_round, "max_round":max_round}

  def get_max_round(self):
    # flag 2
    return self.data["max_round"]

  def get_model_params(self):
    # flag 3
    return {'id': self.id, 'params':list(map(lambda layer: np.array(layer), self.data))}

  def put_model_params(self, params):
    self.data = params

  def get_error_code(self):
    # flag 4
    return {'id': self.id, 'error_code': self.data}

  def put_error_code(self, code):
    self.data = code
  def get_num_data(self):
    # flag 5
    return {'id': self.id, 'num_data': self.data}
  def put_num_data(self, num_data):
    self.data = num_data

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