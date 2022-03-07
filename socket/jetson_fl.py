from json_socket import Client, Message
import tensorflow as tf
import numpy as np

class FLClient:
    def __init__(self, id, host = 'localhost', port = 20000):
        self.id = id
        self.host = host
        self.port = port
        self.sock_client = Client()
        self.sock_client.connect(id, host, port)

        self.model = self.build_model_from_server()
     
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.prepare_dataset(self.get_dataset_name())

    def task(self):
        while True:
            msg = self.recv_msg()
            if msg.flag == Message.TERMINATE:
                break

            if msg.flag == Message.FLAG_GET_PARAMS:
                self.weights = self.get_model_param()

            if msg.flag == Message.FLAG_GET_STATUS_CODE:
                self.send_status_code()

            if msg.flag == Message.FLAG_START_TRAIN:
                global_weight = self.get_model_param()
                self.train_model(global_weight)
                self.send_model_param()
                

    def build_model_from_server(self):
        model_arch = self.get_model_arch()
        #model = tf.keras.models.model_from_json(model, custom_objects={"null":None}) 
        model = tf.keras.models.model_from_json(model_arch)

        data = self.get_compile_config()
        optimizer, loss, metrics = data['optim'], data['loss'], data['metrics']
        model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
        return model
    
    def prepare_dataset(self, name):
        if name == "mnist":
            return tf.keras.datasets.mnist.load_data(path="mnist.npz")
            
        if name == "cifar10":
            return tf.keras.datasets.cifar10.load_data(path="cifar10.npz")

        if name == "cifar100":
            return tf.keras.datasets.cifar100.load_data(path="cifar100.npz")
        
        if name == "imdb":
            return tf.keras.datasets.imdb.load_data(path="imdb.npz")

        if name == "fmnist":
            return tf.keras.datasets.fashion_mnist.load_data(path="fmnist.npz")

    def get_data_idxs(self):
        return self.send_recv_msg(Message.FLAG_GET_DATA_IDX).data['data_idxs']

    def get_dataset_name(self):
        return self.send_recv_msg(Message.FLAG_GET_DATA_NAME).data['dataset_name']

    def get_compile_config(self):
        return self.send_recv_msg(Message.FLAG_GET_CONFIG).data

    def get_model_arch(self):
        return self.send_recv_msg(Message.FLAG_GET_ARCH).data['arch']

    def get_model_param(self):
        return self.send_recv_msg(Message.FLAG_GET_PARAMS).data['param']

    def send_model_param(self):
        return self.send_recv_msg(Message.FLAG_GET_PARAMS, data=list(map(lambda layer: layer.tolist(), self.model.get_weights())))

    
    def health_check(self):
        try:
            global_weight = self.get_model_param()

            if global_weight != None:
                global_weight = list(map(lambda weight: np.array(weight), global_weight))
                self.model.set_weights(global_weight)
            
            test_idxs = np.random.choice(len(self.x_train), 100)
            split_x_train, split_y_train = self.x_train[test_idxs], self.y_train[test_idxs]
            self.model.fit(split_x_train, split_y_train, epochs=1, batch_size=8, verbose=0)
            return Message.HEALTH_GOOD

        except:
            return Message.HEALTH_BAD
            

    def send_status_code(self):
        data = self.health_check()
        return self.send_recv_msg(Message.FLAG_GET_STATUS_CODE, data=data)

    def send_recv_msg(self, flag=Message.FLAG_GET_ARCH, data=None):
        msg = Message(source=self.id, flag=flag, data=data)
        self.sock_client.send(msg)
        response = self.sock_client.recv()
        return response

    def recv_msg(self):
        msg = self.sock_client.recv()
        return msg 

    def train_model(self, global_weight):
        # Train a local model from latest model parameters 
        data_idxs = self.get_data_idxs()

        if global_weight != None:
            global_weight = list(map(lambda weight: np.array(weight), global_weight))
            self.model.set_weights(global_weight)
            
        split_x_train, split_y_train = self.x_train[data_idxs], self.y_train[data_idxs]

        self.model.fit(split_x_train, split_y_train, epochs=10, batch_size=8, verbose=0)
